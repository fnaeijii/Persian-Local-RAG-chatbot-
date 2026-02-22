
# ==================== تنظیمات مدل‌های Embedding ====================
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
FALLBACK_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"
TOKENIZER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ==================== تنظیمات Ollama (نگه داشته شده برای سازگاری) ====================
OLLAMA_MODEL = "deepseek-r1:latest"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 600
OLLAMA_MAX_RETRIES = 3
OLLAMA_RETRY_DELAY = 2

OLLAMA_ADVANCED_OPTIONS = {
    "temperature": 0.3,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_ctx": 32768,
    "mirostat": 2,
    "mirostat_tau": 5.0,
    "seed": 42,
    "stop": ["<|endoftext|>", "###", "سوال:", "Question:", "\n\n\n"]
}

# ==================== پارامترهای Chunking ====================
MAX_TOKENS = 384
OVERLAP_TOKENS = 96
MIN_CHUNK_LENGTH = 100

# ==================== پارامترهای جستجو ====================
DENSE_SEARCH_TOP_K = 100
BM25_SEARCH_TOP_K = 100
RRF_TOP_N = 80
RERANK_TOP_K = 10
RRF_K = 60

SCORE_THRESHOLD = 0.8
MIN_PARAGRAPHS = 1
MAX_PARAGRAPHS = 10
# ==================== مسیرهای فایل ====================
INDEX_BIN_PATH = "faiss.index"
META_PKL_PATH = "faiss_meta.pkl"

# ==================== پردازش متن ====================
SUPPORTED_ENCODINGS = ['utf-8', 'utf-8-sig', 'utf-16', 'windows-1256', 'arabic']

# ==================== تنظیمات سند Word ====================
DEFAULT_FONT = 'B Nazanin'
TITLE_FONT_SIZE = 14
HEADING_FONT_SIZE = 12
BODY_FONT_SIZE = 11
DATE_FONT_SIZE = 10

QUESTION_COLOR = (0, 0, 139)  # Dark blue
ANSWER_COLOR = (0, 128, 0)    # Green
DATE_COLOR = (128, 128, 128)  # Gray



