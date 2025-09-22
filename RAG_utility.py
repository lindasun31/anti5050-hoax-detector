import os
import re
import json
import requests
import pandas as pd
from urllib.parse import urlparse
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import numpy as np
# --- Utility Functions ---

def load_text_file(filepath):
    """Membaca konten dari file teks."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Peringatan: File {filepath} tidak ditemukan.")
        return ""
    except Exception as e:
        print(f"Error saat membaca file {filepath}: {str(e)}")
        return ""


def get_domain(url):
    """Mengekstrak domain dari URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        return domain
    except Exception:
        return "unknown_domain"


def clean_text_content(text):
    """Membersihkan dan menormalkan konten teks."""
    if text is None:
        return "Tidak ada konten yang tersedia."
    text = re.sub(r'[\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def sanitize_content(content):
    """Membersihkan konten untuk pemrosesan dan membatasinya."""
    if content is None or content.strip() == "" or content.strip() == "Tidak ada konten yang tersedia.":
        return "Tidak ada konten yang tersedia."
    content = clean_text_content(content)
    content = content.replace("{", "{{").replace("}", "}}")
    if len(content) > 10000:
        content = content[:10000] + "..."
    return content


def load_json_file(json_filename="domain/front-space_15_domains_no-turnbackhoax.json"):
    """Memuat file JSON dengan penanganan error."""
    try:
        with open(json_filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {json_filename} tidak ditemukan.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Tidak dapat mendekode JSON dari {json_filename}.")
        return None
    except Exception as e:
        print(f"Error saat memuat JSON: {str(e)}")
        return None


# --- Core RAG Components (Parallelized) ---

def generate_queries(userInput, llm):
    """Menghasilkan kueri pencarian dari klaim pengguna."""
    if not userInput or len(userInput.strip()) == 0:
        return ["Input tidak valid"], "Error: Input kosong"
    prompt_query = load_text_file("prompts/query_gen-3questions_one-shot.txt")
    query_prompt = ChatPromptTemplate.from_template(prompt_query)
    query_chain = query_prompt | llm | StrOutputParser()
    try:
        query_text = query_chain.invoke({"query": userInput})
        lines = query_text.strip().split('\n')
        queries = [re.match(r'^\s*\d+\.\s*(.+)$', line).group(1).strip() for line in lines if
                   re.match(r'^\s*\d+\.\s*(.+)$', line)]
        queries = queries[:3] if queries else [userInput[:200]]
        return queries, query_text
    except Exception as e:
        print(f"Error saat menghasilkan kueri: {e}")
        return [userInput[:200]], f"Error saat menghasilkan kueri: {str(e)}"


def search_tavily_parallel(queries, api_key, domain_list_path):
    """Mencari di Tavily menggunakan kueri yang dihasilkan secara paralel."""
    if not api_key:
        print("Error: Kunci API Tavily tidak dikonfigurasi.")
        return {"results": []}

    trusted_domains = load_json_file(domain_list_path)
    if trusted_domains is None:
        print("Peringatan: Gagal memuat domain tepercaya. Melanjutkan tanpa filter domain.")
        trusted_domains = []

    all_search_results = []

    def fetch_query_results(query):
        """Fungsi helper untuk melakukan satu permintaan pencarian Tavily."""
        payload = {
            "query": query, "topic": "general", "search_depth": "basic", "max_results": 10,
            "include_domains": trusted_domains, "include_raw_content": True
        }
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload, timeout=30
            )
            response.raise_for_status()
            search_results = response.json()
            for result in search_results.get('results', []):
                result['source_query'] = query
            return search_results.get('results', [])
        except requests.exceptions.RequestException as e:
            print(f"Error permintaan saat mencari Tavily untuk kueri '{query}': {e}")
            return []

    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        future_to_query = {executor.submit(fetch_query_results, query): query for query in queries}
        for future in as_completed(future_to_query):
            results = future.result()
            if results:
                all_search_results.extend(results)

    return {"results": all_search_results}


def extract_content_from_urls_parallel(urls, api_key):
    """Mengekstrak konten penuh dari URL secara paralel menggunakan Tavily Extract API."""
    if not urls:
        return {}

    if not api_key:
        print("Error: Kunci API Tavily tidak dikonfigurasi untuk ekstraksi.")
        return {}

    url_to_content = {}
    batch_size = 5
    url_batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]

    def fetch_batch(batch_urls):
        """Mengambil konten untuk satu batch URL."""
        payload = {"urls": batch_urls, "include_images": False, "extract_depth": "basic"}
        try:
            response = requests.post(
                "https://api.tavily.com/extract",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload, timeout=60
            )
            if response.status_code == 200:
                return {res.get('url'): res.get('raw_content') for res in response.json().get('results', []) if
                        res.get('url')}
        except requests.exceptions.RequestException as e:
            print(f"Error permintaan saat mengekstrak batch: {e}")
        return {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_batch = {executor.submit(fetch_batch, batch): batch for batch in url_batches}
        for future in as_completed(future_to_batch):
            result_batch = future.result()
            if result_batch:
                url_to_content.update(result_batch)

    print(f"Berhasil mengekstrak konten dari {len(url_to_content)}/{len(urls)} URL")
    return url_to_content


def process_search_results(search_results):
    """Memproses hasil pencarian: deduplikasi, pemeringkatan, dan identifikasi konten yang akan diekstrak."""
    if not search_results or 'results' not in search_results:
        return [], [], []

    sanitized_results = []
    for result in search_results.get('results', []):
        result['domain'] = get_domain(result['url'])
        if 'content' in result and result['content'] is not None:
            result['content'] = sanitize_content(result['content'])
        result['extracted_content'] = None
        sanitized_results.append(result)

    query_results = {}
    for item in sanitized_results:
        query = item['source_query']
        query_results.setdefault(query, []).append(item)

    final_results = []
    for query, results in query_results.items():
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        final_results.extend(sorted_results[:5])

    unique_results = {item['url']: item for item in final_results}
    deduped_results = list(unique_results.values())

    original_raw_content_map = {item['url']: item.get('raw_content') for item in search_results.get('results', []) if
                                item.get('url')}
    urls_to_extract = [
        item['url'] for item in deduped_results if not (original_raw_content_map.get(item['url']) or "").strip()
    ]

    return deduped_results, urls_to_extract, list(query_results.keys())


def get_content_and_finalize(deduped_results, urls_to_extract, queries_used, tavily_api_key):
    """Mengekstrak konten jika diperlukan dan menyelesaikan pemrosesan hasil."""
    url_to_extracted_content = {}
    if urls_to_extract:
        urls_to_extract = list(set(urls_to_extract))
        print(f"Mencoba mengekstrak konten untuk {len(urls_to_extract)} hasil...")
        url_to_extracted_content = extract_content_from_urls_parallel(urls_to_extract, tavily_api_key)

    for result in deduped_results:
        url = result.get('url')
        if url in url_to_extracted_content:
            result['extracted_content'] = sanitize_content(url_to_extracted_content[url])
        if 'raw_content' in result and result.get('raw_content'):
            result['raw_content'] = sanitize_content(result['raw_content'])

    return {
        "results": deduped_results,
        "stats": {
            "total_processed_results": len(deduped_results),
            "urls_extracted_count": len(url_to_extracted_content),
            "queries_used": queries_used
        }
    }


def generate_fact_check_analysis(user_query, processed_results, llm):
    """Menghasilkan analisis pengecekan fakta berdasarkan bukti."""
    evidence_results = processed_results.get('results', [])
    system_prompt = load_text_file("prompts/system_prompt2.txt")

    user_prompt_text = f"Klaim yang perlu di-fact check: {user_query}\nSumber-sumber bukti yang tersedia:\n"
    for i, evidence in enumerate(evidence_results, 1):
        content = (
                evidence.get('extracted_content') or evidence.get('raw_content') or
                evidence.get('content') or "Tidak ada konten yang tersedia."
        )
        user_prompt_text += f"""
BUKTI {i}:
Judul: {evidence.get('title', 'N/A')}
URL: {evidence.get('url', 'N/A')}
Konten: {sanitize_content(content)}
---
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt_text)
    ])
    generation_chain = prompt | llm | StrOutputParser()
    try:
        analysis = generation_chain.invoke({})
        return {"analysis": analysis, "status": "success", "full_prompt": user_prompt_text}
    except Exception as e:
        error_message = f"Error saat menghasilkan analisis: {str(e)}"
        print(error_message)
        return {"analysis": error_message, "status": "error"}


# --- Main Pipeline Orchestrator ---

def process_rag_pipeline_parallel(user_input_text, openrouter_api_key, tavily_api_key):
    """Menjalankan seluruh pipeline RAG dengan pemrosesan paralel."""
    results = {}
    start_time = time.time()

    # Inisialisasi LLM di dalam fungsi untuk memastikan kunci API digunakan dengan benar
    llm = ChatOpenAI(
        model="google/gemini-2.0-flash-001",
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

    try:
        # Langkah 1: Input Pengguna
        results["user_input"] = user_input_text
        # print(f"1. Memproses klaim: {user_input_text}")

        # Langkah 2: Generasi Kueri
        queries, _ = generate_queries(user_input_text, llm)
        results["queries"] = queries
        # print(f"2. Kueri yang dihasilkan: {queries}")

        # Langkah 3: Pencarian Web Paralel
        search_results = search_tavily_parallel(queries, tavily_api_key,
                                                "domain/15domain.json")
        results["search_results_count"] = len(search_results.get('results', []))
        # print(f"3. Ditemukan {results['search_results_count']} hasil pencarian awal.")

        # Langkah 4: Proses Hasil Awal
        deduped_results, urls_to_extract, queries_used = process_search_results(search_results)
        # print(
        #     f"4. Hasil diproses menjadi {len(deduped_results)} item unik. {len(urls_to_extract)} URL memerlukan ekstraksi konten.")

        # Langkah 5: Ekstraksi Konten Paralel & Finalisasi
        processed_results = get_content_and_finalize(deduped_results, urls_to_extract, queries_used, tavily_api_key)
        results["processed_results"] = processed_results
        # print("5. Ekstraksi konten selesai.")

        # Langkah 6: Analisis Pengecekan Fakta
        fact_check_result = generate_fact_check_analysis(user_input_text, processed_results, llm)
        results["fact_check_analysis"] = fact_check_result
        # print("6. Analisis pengecekan fakta selesai.")

        end_time = time.time()
        results["total_time"] = round(end_time - start_time, 2)
        print(f"Pipeline selesai dalam {results['total_time']} detik.")

        return results, None

    except Exception as e:
        print(f"Error pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return results, str(e)

def analysis_separator(analysis_text, relevance_scores):
    """Memisahkan kesimpulan dan daftar bukti dari teks analisis."""
    # Ambil kesimpulan
    conclusion_match = re.search(r"KESIMPULAN:\s*(\w+)", analysis_text)
    conclusion = conclusion_match.group(1) if conclusion_match else None

    # Pattern untuk bukti
    bukti_pattern = re.compile(r"\d+\.\s*([^:]+):\s*([^-]+)-\s*(.*?)\s*-\s*(https?://\S+)", re.DOTALL)
    bukti_list = []

    # Loop untuk setiap bukti
    for i, match in enumerate(bukti_pattern.finditer(analysis_text)):
        # Pastikan indeks relevansi ada
        if i >= len(relevance_scores):
            relevance_score = 0
        else:
            relevance_score = relevance_scores[i]

        if relevance_score > 0.5:  # Hanya ambil relevansi > 0.5
            domain = match.group(1).strip()
            label = match.group(2).strip()
            title = match.group(3).strip()
            url = match.group(4).strip()

            bukti_list.append({
                "domain": domain,
                "label": label,
                "title": title,
                "url": url,
                "relevance": relevance_score
            })

    parsed_result = {
        "kesimpulan": conclusion,
        "bukti": bukti_list
    }

    return parsed_result


def confidence_calculator(parsed_result):
    """Hitung confidence level dari daftar bukti."""
    bukti = parsed_result.get("bukti", [])

    if len(bukti) == 0:
        return 0, 0, 0.0  # Tidak ada bukti → return 0

    multiplier = []
    relevance_score = []
    support = 0
    not_support = 0

    for item in bukti:
        label = item.get("label", "").lower()
        relevance = item.get("relevance", 0)
        relevance_score.append(relevance)

        if label == "mendukung":
            multiplier.append(1)
            support += 1
        elif label == "tidak mendukung" or label == "membantah":
            multiplier.append(-1)
            not_support += 1
        else:  # Bisa tambahkan "tidak relevan" → set 0
            multiplier.append(0)

    multiplier = np.array(multiplier)
    relevance_score = np.array(relevance_score)

    # Cegah pembagian dengan 0
    if np.sum(relevance_score) == 0:
        confidence = 0.0
    else:
        confidence = np.abs(np.round((np.dot(multiplier, relevance_score) / np.sum(relevance_score)) * 100, 2))

    return support, not_support, confidence
