/* SIDPOL · asistente de IA (sin JavaScript en línea, compatible con la CSP con nonce) */
(function () {
    'use strict';

    // Escapa HTML antes de insertar texto en la página (prevención de XSS)
    function escapeHtml(t) {
        return String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }
    function formatText(text) {
        return escapeHtml(text).replace(/\*\*(.*?)\*\*/g, '<b>$1</b>').replace(/\* /g, '<br>• ').replace(/\n/g, '<br>');
    }

    function burbuja(chatBody, html, clase, lado) {
        const fila = document.createElement('div');
        fila.className = 'd-flex mb-3 justify-content-' + lado;
        fila.innerHTML = `<div class="${clase} shadow-sm">${html}</div>`;
        chatBody.appendChild(fila);
        chatBody.scrollTop = chatBody.scrollHeight;
        return fila;
    }

    function toggleChat() {
        const w = document.getElementById('chatbot-window');
        const oculto = w.style.display === 'none';
        w.style.display = oculto ? 'flex' : 'none';
        if (oculto) document.getElementById('chat-input').focus();
    }

    function sendMessage() {
        const input = document.getElementById('chat-input');
        const msg = input.value.trim();
        if (!msg) return;
        const chatBody = document.getElementById('chat-messages');
        burbuja(chatBody, escapeHtml(msg), 'chat-bubble-user', 'end');
        input.value = '';
        const cargando = burbuja(chatBody, '<span></span><span></span><span></span>', 'chat-bubble-bot typing', 'start');

        fetch('/chat-ia', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': document.querySelector('meta[name="csrf-token"]').content
            },
            body: 'mensaje=' + encodeURIComponent(msg)
        })
            .then(r => r.json())
            .then(data => {
                cargando.remove();
                let html = formatText(data.respuesta || 'Sin respuesta.');
                const fuentes = (data.fuentes || []).filter(Boolean);
                if (fuentes.length) {
                    html += `<div class="sp-chat-fuentes"><i class="fas fa-database me-1"></i>${escapeHtml(fuentes.join(' · '))}</div>`;
                }
                burbuja(chatBody, html, 'chat-bubble-bot', 'start');
            })
            .catch(() => {
                cargando.remove();
                burbuja(chatBody, 'No se pudo conectar con el asistente.', 'chat-bubble-bot', 'start');
            });
    }

    document.addEventListener('DOMContentLoaded', function () {
        document.querySelectorAll('[data-chat="alternar"]').forEach(b => b.addEventListener('click', toggleChat));
        const enviar = document.getElementById('chat-enviar');
        if (enviar) enviar.addEventListener('click', sendMessage);
        const input = document.getElementById('chat-input');
        if (input) input.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });
    });
})();
