import http.server
import socketserver
import webbrowser
import json
import urllib.parse
import os
import cgi
from datetime import datetime

# Global variables
current_file = None
index_built = False
uploaded_pdf_path = None  # Store original PDF path


class RAGHandler(http.server.BaseHTTPRequestHandler):

    def _set_json_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_OPTIONS(self):
        # Allow CORS preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            return

        if parsed.path == '/download':
            qs = urllib.parse.parse_qs(parsed.query)
            filename = qs.get('file', [None])[0]
            if filename and os.path.exists(filename):
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(filename)}"')
                    self.end_headers()
                    with open(filename, 'rb') as f:
                        self.wfile.write(f.read())
                except Exception as e:
                    self.send_error(500, f'Error serving file: {e}')
            else:
                self.send_error(404, 'File not found')
            return

        # Unknown GET
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        try:
            if self.path == '/upload':
                self.handle_upload()
            elif self.path == '/index':
                self.handle_index()
            elif self.path == '/ask':
                self.handle_ask()
            else:
                self.send_error_response('Invalid endpoint')
        except Exception as e:
            self.send_error_response(f'خطا: {str(e)}')

    def pdf_to_text(self, pdf_path):
        """Convert PDF to text using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                if page_text.strip():
                    text += f"\n--- صفحه {page_num} ---\n"
                    text += page_text
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"خطا در تبدیل PDF: {str(e)}")

    def handle_upload(self):
        global uploaded_pdf_path

        # Use cgi to parse multipart form data
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST'})

        if 'file' not in form:
            self.send_json_response({'success': False, 'message': 'فایلی ارسال نشده است'})
            return

        fileitem = form['file']
        if not fileitem.filename:
            self.send_json_response({'success': False, 'message': 'فایل نامعتبر است'})
            return

        filename = os.path.basename(fileitem.filename)

        # Accept both PDF and TXT files
        if not (filename.lower().endswith('.pdf') or filename.lower().endswith('.txt')):
            self.send_json_response({'success': False, 'message': 'فقط فایل‌های PDF و TXT پشتیبانی می‌شوند'})
            return

        try:
            # Save uploaded file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f'temp_{timestamp}_{filename}'

            with open(filepath, 'wb') as f:
                data = fileitem.file.read()
                f.write(data)

            uploaded_pdf_path = filepath

            # Get file info
            file_size = os.path.getsize(filepath) / 1024  # KB

            # If PDF, get page count
            page_count = 0
            if filename.lower().endswith('.pdf'):
                try:
                    doc = fitz.open(filepath)
                    page_count = len(doc)
                    doc.close()
                except:
                    page_count = 0

            self.send_json_response({
                'success': True,
                'message': f'فایل {filename} با موفقیت آپلود شد',
                'filename': filename,
                'size': f'{file_size:.1f} KB',
                'pages': page_count if page_count > 0 else None
            })

        except Exception as e:
            self.send_json_response({'success': False, 'message': f'خطا در بارگذاری: {str(e)}'})

    def handle_index(self):
        global current_file, index_built, uploaded_pdf_path

        if not uploaded_pdf_path or not os.path.exists(uploaded_pdf_path):
            self.send_json_response({'success': False, 'message': 'ابتدا فایل را آپلود کنید'})
            return

        try:
            # Remove old index files if exist
            try:
                if os.path.exists(config.INDEX_BIN_PATH):
                    os.remove(config.INDEX_BIN_PATH)
                if os.path.exists(config.META_PKL_PATH):
                    os.remove(config.META_PKL_PATH)
            except Exception:
                pass

            # Convert to text if PDF
            if uploaded_pdf_path.lower().endswith('.pdf'):
                # Convert PDF to text
                text_content = self.pdf_to_text(uploaded_pdf_path)

                # Save as temporary text file
                txt_filepath = uploaded_pdf_path.replace('.pdf', '.txt').replace('.PDF', '.txt')
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(text_content)

                # Build index with text file
                build_index(txt_filepath)
                current_file = txt_filepath
            else:
                # Already a text file
                build_index(uploaded_pdf_path)
                current_file = uploaded_pdf_path

            index_built = True

            self.send_json_response({
                'success': True,
                'message': 'ایندکس با موفقیت ساخته شد'
            })

        except Exception as e:
            self.send_json_response({'success': False, 'message': f'خطا در ساخت ایندکس: {str(e)}'})

    def handle_ask(self):
        global index_built

        # Read content length and body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        if not index_built:
            self.send_json_response({'success': False, 'message': 'ابتدا ایندکس را بسازید'})
            return

        try:
            data = json.loads(body.decode('utf-8'))
            question = data.get('question', '').strip()

            if not question:
                self.send_json_response({'success': False, 'message': 'سوال نمی‌تواند خالی باشد'})
                return

            # Search
            results = search_by_threshold(question)

            # Format results
            paragraphs = []
            top_paragraphs = []

            for para, score in results:
                paragraphs.append({
                    'text': para,
                    'score': f'{score:.4f}'
                })
                top_paragraphs.append(para)

            # Get answer from AI
            answer = ask_deepseek(top_paragraphs, question)


            self.send_json_response({
                'success': True,
                'paragraphs': paragraphs,
                'answer': answer,

            })

        except Exception as e:
            self.send_json_response({'success': False, 'message': f'خطا در پردازش سوال: {str(e)}'})

    def send_json_response(self, data):
        response = json.dumps(data, ensure_ascii=False)
        self._set_json_headers()
        self.wfile.write(response.encode('utf-8'))

    def send_error_response(self, message):
        self.send_json_response({'success': False, 'message': message})


# HTML Template with Persian UI and specified colors
HTML_TEMPLATE = '''<!DOCTYPE html>
<html dir="rtl" lang="fa">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>سیستم پرسش و پاسخ هوشمند</title>
    <link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #222831;
            --panel: #393E46;
            --button: #DFD0B8;
            --text: #948979;
            --white: #ffffff;
            --success: #4ade80;
            --error: #f87171;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Vazirmatn', Tahoma, sans-serif;
            background: var(--bg-dark);
            color: var(--white);
            min-height: 100vh;
        }

        .container {
            display: flex;
            height: 100vh;
        }

        /* Left Panel - PDF Upload & Viewer */
        .left-panel {
            flex: 1;
            background: var(--panel);
            padding: 20px;
            border-left: 1px solid rgba(255,255,255,0.1);
            display: flex;
            flex-direction: column;
        }

        /* Right Panel - Chat Interface */
        .right-panel {
            flex: 1;
            background: var(--bg-dark);
            padding: 20px;
            display: flex;
            flex-direction: column;
        }

        h2 {
            color: var(--white);
            margin-bottom: 20px;
            font-size: 1.5rem;
        }

        /* Upload Area */
        .upload-area {
            background: rgba(0,0,0,0.2);
            border: 2px dashed var(--text);
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }

        .upload-area:hover {
            border-color: var(--button);
            background: rgba(0,0,0,0.3);
        }

        .upload-area.dragover {
            border-color: var(--button);
            background: rgba(223, 208, 184, 0.1);
        }

        input[type="file"] {
            display: none;
        }

        .btn {
            background: var(--button);
            color: var(--bg-dark);
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            font-family: 'Vazirmatn', sans-serif;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(223, 208, 184, 0.3);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .btn-secondary {
            background: transparent;
            border: 2px solid var(--text);
            color: var(--text);
        }

        .btn-secondary:hover {
            border-color: var(--button);
            color: var(--button);
        }

        /* PDF Viewer */
        .pdf-viewer {
            flex: 1;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            margin: 20px 0;
            padding: 20px;
            overflow-y: auto;
        }

        /* Chat Interface */
        .chat-messages {
            flex: 1;
            background: var(--panel);
            border-radius: 10px;
            padding: 20px;
            overflow-y: auto;
            margin-bottom: 20px;
        }

        .message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 10px;
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            background: rgba(223, 208, 184, 0.2);
            border-right: 4px solid var(--button);
        }

        .message.assistant {
            background: rgba(0,0,0,0.2);
            border-right: 4px solid var(--text);
        }

        .message-header {
            font-weight: 700;
            margin-bottom: 10px;
            color: var(--button);
        }

        .message.assistant .message-header {
            color: var(--text);
        }

        /* Chat Input */
        .chat-input-area {
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            background: var(--panel);
            border: 1px solid var(--text);
            color: var(--white);
            padding: 15px;
            border-radius: 10px;
            font-size: 1rem;
            font-family: 'Vazirmatn', sans-serif;
        }

        .chat-input:focus {
            outline: none;
            border-color: var(--button);
        }

        /* Status Messages */
        .status {
            padding: 10px 15px;
            border-radius: 8px;
            margin: 10px 0;
            animation: slideIn 0.3s;
        }

        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        .status.success {
            background: rgba(74, 222, 128, 0.2);
            color: var(--success);
            border-right: 4px solid var(--success);
        }

        .status.error {
            background: rgba(248, 113, 113, 0.2);
            color: var(--error);
            border-right: 4px solid var(--error);
        }

        .status.loading {
            background: rgba(223, 208, 184, 0.2);
            color: var(--button);
            border-right: 4px solid var(--button);
        }

        /* Loader */
        .loader {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid var(--text);
            border-top-color: var(--button);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* File Info */
        .file-info {
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }

        .file-info-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            color: var(--text);
        }

        .file-info-value {
            color: var(--white);
            font-weight: 500;
        }

        /* Paragraphs */
        .paragraph-item {
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-right: 3px solid var(--text);
        }

        .paragraph-score {
            float: left;
            background: var(--button);
            color: var(--bg-dark);
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.9rem;
        }

        /* Download Link */
        .download-link {
            display: inline-block;
            background: var(--button);
            color: var(--bg-dark);
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 700;
            margin-top: 10px;
            transition: all 0.3s;
        }

        .download-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(223, 208, 184, 0.3);
        }

        /* Responsive */
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
            }

            .left-panel, .right-panel {
                flex: none;
                height: 50vh;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Left Panel -->
        <div class="left-panel">
            <h1>📄GPT مخصوص آتش نشانان  </h1>
            <p></p>

            <div id="uploadArea" class="upload-area">
                <svg width="48" height="48" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="margin: 0 auto; color: var(--text);">
                    <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
                <p style="margin: 15px 0; color: var(--text);">فایل PDF یا متنی را اینجا رها کنید</p>
                <input type="file" id="fileInput" accept=".pdf,.txt">
                <label for="fileInput" class="btn">انتخاب فایل</label>
            </div>

            <div id="fileInfo" class="file-info" style="display: none;">
                <div class="file-info-item">
                    <span>نام فایل:</span>
                    <span class="file-info-value" id="fileName">-</span>
                </div>
                <div class="file-info-item">
                    <span>حجم فایل:</span>
                    <span class="file-info-value" id="fileSize">-</span>
                </div>
                <div class="file-info-item" id="pageCountRow" style="display: none;">
                    <span>تعداد صفحات:</span>
                    <span class="file-info-value" id="pageCount">-</span>
                </div>
            </div>

            <div style="display: flex; gap: 10px; margin: 20px 0;">
                <button id="indexBtn" class="btn" onclick="buildIndex()" disabled>ساخت ایندکس</button>
                <button class="btn btn-secondary" onclick="clearAll()">پاک کردن</button>
            </div>

            <div id="uploadStatus"></div>

            <div class="pdf-viewer">
                <div id="pdfContent" style="color: var(--text); text-align: center;">
                    فایلی بارگذاری نشده است
                </div>
            </div>
        </div>

        <!-- Right Panel -->
        <div class="right-panel">
            <h2>💬 پرسش و پاسخ</h2>

            <div class="chat-messages" id="chatMessages">
                <div style="text-align: center; color: var(--text); padding: 50px;">
                    ابتدا فایل را آپلود کرده و ایندکس بسازید، سپس سوال خود را بپرسید
                </div>
            </div>

            <div class="chat-input-area">
                <input type="text" id="questionInput" class="chat-input" 
                       placeholder="سوال خود را اینجا بنویسید..." disabled
                       onkeypress="if(event.key==='Enter') askQuestion()">
                <button id="askBtn" class="btn" onclick="askQuestion()" disabled>ارسال</button>
            </div>
        </div>
    </div>

<script>
    let uploadedFile = null;
    let indexReady = false;

    // Drag and Drop
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file);
    });

    function handleFile(file) {
        uploadedFile = file;
        uploadFile();
    }

    function showStatus(elementId, message, type = 'info') {
        const element = document.getElementById(elementId);
        element.innerHTML = `<div class="status ${type}">${message}</div>`;

        if (type !== 'loading') {
            setTimeout(() => {
                element.innerHTML = '';
            }, 5000);
        }
    }

    async function uploadFile() {
        if (!uploadedFile) {
            showStatus('uploadStatus', 'لطفاً فایلی انتخاب کنید', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', uploadedFile);

        showStatus('uploadStatus', 'در حال آپلود فایل... <span class="loader"></span>', 'loading');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Show file info
                document.getElementById('fileInfo').style.display = 'block';
                document.getElementById('fileName').textContent = data.filename;
                document.getElementById('fileSize').textContent = data.size;

                if (data.pages) {
                    document.getElementById('pageCountRow').style.display = 'flex';
                    document.getElementById('pageCount').textContent = data.pages + ' صفحه';
                }

                // Enable index button
                document.getElementById('indexBtn').disabled = false;

                // Show success message
                showStatus('uploadStatus', data.message, 'success');

                // Update PDF viewer
                document.getElementById('pdfContent').innerHTML = 
                    `<p style="color: var(--button);">✓ فایل آماده ساخت ایندکس است</p>`;
            } else {
                showStatus('uploadStatus', data.message, 'error');
            }
        } catch (error) {
            showStatus('uploadStatus', 'خطا در ارتباط با سرور', 'error');
        }
    }

    async function buildIndex() {
        const btn = document.getElementById('indexBtn');
        btn.disabled = true;
        btn.innerHTML = 'در حال ساخت ایندکس... <span class="loader"></span>';

        showStatus('uploadStatus', 'در حال ساخت ایندکس، لطفاً صبر کنید...', 'loading');

        try {
            const response = await fetch('/index', {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                indexReady = true;

                // Enable chat
                document.getElementById('questionInput').disabled = false;
                document.getElementById('askBtn').disabled = false;

                // Update status
                showStatus('uploadStatus', data.message, 'success');

                // Update PDF viewer
                document.getElementById('pdfContent').innerHTML = 
                    `<p style="color: var(--success);">✓ ایندکس آماده است</p>
                     <p style="color: var(--text); margin-top: 10px;">می‌توانید سوالات خود را بپرسید</p>`;

                // Clear chat and show ready message
                document.getElementById('chatMessages').innerHTML = 
                    `<div class="message assistant">
                        <div class="message-header">🤖 دستیار هوشمند</div>
                        <div>سلام! ایندکس فایل شما آماده است. سوال خود را بپرسید.</div>
                    </div>`;

                // Focus on input
                document.getElementById('questionInput').focus();
            } else {
                showStatus('uploadStatus', data.message, 'error');
            }
        } catch (error) {
            showStatus('uploadStatus', 'خطا در ساخت ایندکس', 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = 'ساخت ایندکس';
        }
    }

    async function askQuestion() {
        const input = document.getElementById('questionInput');
        const question = input.value.trim();

        if (!question) {
            showStatus('uploadStatus', 'لطفاً سوال خود را وارد کنید', 'error');
            return;
        }

        if (!indexReady) {
            showStatus('uploadStatus', 'ابتدا ایندکس را بسازید', 'error');
            return;
        }

        // Add user message to chat
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML += `
            <div class="message user">
                <div class="message-header">👤 شما</div>
                <div>${question}</div>
            </div>
        `;

        // Clear input
        input.value = '';

        // Disable input during processing
        input.disabled = true;
        document.getElementById('askBtn').disabled = true;

        // Add loading message
        chatMessages.innerHTML += `
            <div class="message assistant" id="loadingMessage">
                <div class="message-header">🤖 دستیار هوشمند</div>
                <div>در حال پردازش... <span class="loader"></span></div>
            </div>
        `;

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question })
            });

            const data = await response.json();

            // Remove loading message
            document.getElementById('loadingMessage').remove();

            if (data.success) {
                // Format paragraphs HTML
                let paragraphsHtml = '<div style="margin: 15px 0;"><strong>پاراگراف‌های مرتبط:</strong></div>';
                data.paragraphs.forEach(p => {
                    paragraphsHtml += `
                        <div class="paragraph-item">
                            <span class="paragraph-score">${p.score}</span>
                            <div>${p.text}</div>
                        </div>
                    `;
                });

                // Add assistant response
                chatMessages.innerHTML += `
                    <div class="message assistant">
                        <div class="message-header">🤖 دستیار هوشمند</div>
                        <div>
                            ${paragraphsHtml}
                            <div style="margin: 20px 0; padding: 15px; background: rgba(223, 208, 184, 0.1); border-radius: 8px;">
                                <strong>پاسخ:</strong>
                                <div style="margin-top: 10px; white-space: pre-wrap;">${data.answer}</div>
                            </div>
                        </div>
                    </div>
                `;
            } else {
                // Show error message
                chatMessages.innerHTML += `
                    <div class="message assistant">
                        <div class="message-header">🤖 دستیار هوشمند</div>
                        <div style="color: var(--error);">⚠️ ${data.message}</div>
                    </div>
                `;
            }

        } catch (error) {
            // Remove loading message if still exists
            const loadingMsg = document.getElementById('loadingMessage');
            if (loadingMsg) loadingMsg.remove();

            // Show error message
            chatMessages.innerHTML += `
                <div class="message assistant">
                    <div class="message-header">🤖 دستیار هوشمند</div>
                    <div style="color: var(--error);">⚠️ خطا در ارتباط با سرور</div>
                </div>
            `;
        } finally {
            // Re-enable input
            input.disabled = false;
            document.getElementById('askBtn').disabled = false;
            input.focus();

            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    function clearAll() {
        // Reset everything
        uploadedFile = null;
        indexReady = false;

        // Clear file input
        document.getElementById('fileInput').value = '';

        // Hide file info
        document.getElementById('fileInfo').style.display = 'none';

        // Disable buttons
        document.getElementById('indexBtn').disabled = true;
        document.getElementById('questionInput').disabled = true;
        document.getElementById('askBtn').disabled = true;

        // Clear status
        document.getElementById('uploadStatus').innerHTML = '';

        // Reset PDF viewer
        document.getElementById('pdfContent').innerHTML = 
            '<div style="color: var(--text); text-align: center;">فایلی بارگذاری نشده است</div>';

        // Reset chat
        document.getElementById('chatMessages').innerHTML = 
            '<div style="text-align: center; color: var(--text); padding: 50px;">ابتدا فایل را آپلود کرده و ایندکس بسازید، سپس سوال خود را بپرسید</div>';

        // Clear input
        document.getElementById('questionInput').value = '';

        // Show success message
        showStatus('uploadStatus', 'همه داده‌ها پاک شد', 'success');
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+O to open file dialog
        if (e.ctrlKey && e.key === 'o') {
            e.preventDefault();
            document.getElementById('fileInput').click();
        }

        // Ctrl+Enter to send question
        if (e.ctrlKey && e.key === 'Enter') {
            if (!document.getElementById('askBtn').disabled) {
                askQuestion();
            }
        }

        // Escape to clear
        if (e.key === 'Escape') {
            if (confirm('آیا می‌خواهید همه داده‌ها را پاک کنید؟')) {
                clearAll();
            }
        }
    });

    // Auto-resize chat input
    const questionInput = document.getElementById('questionInput');
    questionInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 150) + 'px';
    });

    // Show initial help tooltip
    window.addEventListener('load', () => {
        setTimeout(() => {
            showStatus('uploadStatus', '💡 می‌توانید فایل PDF یا متنی را با کشیدن و رها کردن یا کلیک آپلود کنید', 'success');
        }, 1000);
    });
</script>
</body>
</html>
'''


def start_server():
    PORT = 8000

    # Create uploads directory if not exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Use ThreadingTCPServer for better performance
    with socketserver.ThreadingTCPServer(("", PORT), RAGHandler) as httpd:
        print(f"\n{'=' * 50}")
        print(f"🚀 سرور RAG فارسی در حال اجرا")
        print(f"{'=' * 50}")
        print(f"📍 آدرس محلی: http://localhost:{PORT}")
        print(f"📍 آدرس شبکه: http://{get_ip_address()}:{PORT}")
        print(f"\n📌 راهنما:")
        print(f"   • فایل PDF یا TXT آپلود کنید")
        print(f"   • دکمه 'ساخت ایندکس' را بزنید")
        print(f"   • سوالات خود را بپرسید")
        print(f"\n⌨️  میانبرها:")
        print(f"   • Ctrl+O: باز کردن فایل")
        print(f"   • Ctrl+Enter: ارسال سوال")
        print(f"   • Escape: پاک کردن همه")
        print(f"\n🛑 برای خروج Ctrl+C را فشار دهید")
        print(f"{'=' * 50}\n")

        # Open browser automatically
        webbrowser.open(f'http://localhost:{PORT}')

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 سرور متوقف شد")
            httpd.shutdown()


def get_ip_address():
    """Get local IP address"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


if __name__ == "__main__":
    # Check required modules
    try:
        import fitz
    except ImportError:
        print("⚠️  کتابخانه PyMuPDF نصب نیست!")
        print("📦 برای نصب:")
        print("   pip install PyMuPDF")
        exit(1)

    try:
        from rag_app import build_index, search_by_threshold, ask_deepseek
        import config
    except ImportError as e:
        print(f"⚠️  خطا در import ماژول‌های RAG: {e}")
        print("📌 مطمئن شوید فایل‌های rag_app.py و config.py در کنار app.py قرار دارند")
        exit(1)

    start_server()