// static/js/main.js

document.addEventListener("DOMContentLoaded", function () {
    // Selector para ir al detalle de modalidad
    const selectorDetalle = document.getElementById("selectorModalidadDetalle");
    if (selectorDetalle) {
        selectorDetalle.addEventListener("change", function () {
            const modalidad = this.value;
            if (modalidad) {
                window.location.href = `/modalidad-detalle/${encodeURIComponent(modalidad)}`;
            }
        });
    }

    // Selector para ir a la predicción de modalidad
    const selectorPred = document.getElementById("selectorModalidadPrediccion");
    if (selectorPred) {
        selectorPred.addEventListener("change", function () {
            const modalidad = this.value;
            if (modalidad) {
                window.location.href = `/prediccion-modalidad/${encodeURIComponent(modalidad)}`;
            }
        });
    }
});

// --- LÓGICA DEL CHATBOT ---

function toggleChat() {
    const windowChat = document.getElementById('chatbot-window');
    const isActive = windowChat.classList.contains('active');
    
    if (isActive) {
        windowChat.classList.remove('active');
        // Pequeño delay para display none si quisieras animar salidas complejas
    } else {
        // Necesitamos asegurar que el display sea block/flex antes de la opacidad
        windowChat.style.display = 'flex';
        // Timeout pequeño para permitir que el navegador procese el cambio de display antes de animar
        setTimeout(() => {
            windowChat.classList.add('active');
        }, 10);
    }
}

function handleEnter(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
}

function enviarTextoDirecto(texto) {
    document.getElementById('chat-input').value = texto;
    sendMessage();
}

function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    const chatBody = document.getElementById('chat-messages');

    if (message === "") return;

    // 1. Agregar mensaje del usuario
    const userBubble = `
        <div class="d-flex flex-row justify-content-end mb-3">
            <div class="message-user shadow-sm">
                <p class="small mb-0">${message}</p>
            </div>
        </div>`;
    chatBody.insertAdjacentHTML('beforeend', userBubble);
    input.value = "";
    chatBody.scrollTop = chatBody.scrollHeight;

    // 2. Mostrar indicador de "Escribiendo..."
    const loadingId = "loading-" + Date.now();
    const loadingBubble = `
        <div id="${loadingId}" class="d-flex flex-row justify-content-start mb-3">
            <div class="message-bot shadow-sm">
                <p class="small mb-0"><i class="fas fa-circle-notch fa-spin"></i> Analizando...</p>
            </div>
        </div>`;
    chatBody.insertAdjacentHTML('beforeend', loadingBubble);
    chatBody.scrollTop = chatBody.scrollHeight;

    // 3. Enviar al Backend (AJAX/Fetch)
    // Asegúrate que tu ruta en Flask sea '/chat-ia' y acepte POST
    fetch('/chat-ia', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `mensaje=${encodeURIComponent(message)}`
    })
    .then(response => response.json()) // Asumiendo que Flask devuelve JSON {respuesta: "..."}
    .then(data => {
        // Remover loading
        document.getElementById(loadingId).remove();

        // Agregar respuesta del Bot (Formateando Markdown si es necesario o texto plano)
        // Convertimos saltos de línea a <br> para HTML básico
        const botText = data.respuesta ? data.respuesta.replace(/\n/g, '<br>') : "Error al procesar.";

        const botBubble = `
            <div class="d-flex flex-row justify-content-start mb-3">
                <div class="p-0"> <div class="message-bot shadow-sm">
                        <p class="small mb-0">${botText}</p>
                    </div>
                </div>
            </div>`;
        
        chatBody.insertAdjacentHTML('beforeend', botBubble);
        chatBody.scrollTop = chatBody.scrollHeight;
    })
    .catch(error => {
        document.getElementById(loadingId).remove();
        console.error('Error:', error);
        // Mensaje de error visual
        const errorBubble = `<div class="text-center text-danger small mb-2">Error de conexión con IA</div>`;
        chatBody.insertAdjacentHTML('beforeend', errorBubble);
    });
}

function reiniciarChat() {
    document.getElementById('chat-messages').innerHTML = `
        <div class="d-flex flex-row justify-content-start mb-3">
            <div class="p-3 bg-light rounded-right rounded-bottom" style="border-top-left-radius: 0;">
                <p class="small mb-0">Chat reiniciado. ¿En qué puedo ayudarte?</p>
            </div>
        </div>
        <div class="quick-actions mb-3 text-center">
            <button class="btn btn-outline-primary btn-sm rounded-pill mb-2" onclick="enviarTextoDirecto('Resumen de riesgos')">Resumen de riesgos</button>
            <button class="btn btn-outline-primary btn-sm rounded-pill mb-2" onclick="enviarTextoDirecto('Predicción 2026')">Predicción 2026</button>
        </div>
    `;
}