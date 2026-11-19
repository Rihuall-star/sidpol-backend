/* SIDPOL · asistente de IA
 * Sin JavaScript en línea (compatible con la CSP con nonce) y sin insertar HTML: todo el texto del
 * usuario y de la IA se inserta con textContent, así ningún contenido puede convertirse en código (XSS). */
(function () {
    'use strict';

    // Convierte el texto de la IA en nodos: **negrita**, viñetas "* " y saltos de línea, sin interpretar HTML
    function textoConFormato(texto) {
        const fragmento = document.createDocumentFragment();
        String(texto).split('\n').forEach(function (linea, i) {
            if (i > 0) fragmento.appendChild(document.createElement('br'));
            linea = linea.replace(/^\s*\* /, '• ');
            linea.split(/(\*\*[^*]+\*\*)/).forEach(function (parte) {
                if (/^\*\*[^*]+\*\*$/.test(parte)) {
                    const b = document.createElement('b');
                    b.textContent = parte.slice(2, -2);
                    fragmento.appendChild(b);
                } else if (parte) {
                    fragmento.appendChild(document.createTextNode(parte));
                }
            });
        });
        return fragmento;
    }

    function burbuja(chatBody, contenido, clase, lado) {
        const fila = document.createElement('div');
        fila.className = 'd-flex mb-3 justify-content-' + lado;
        const caja = document.createElement('div');
        caja.className = clase + ' shadow-sm';
        if (contenido) caja.appendChild(contenido);
        fila.appendChild(caja);
        chatBody.appendChild(fila);
        chatBody.scrollTop = chatBody.scrollHeight;
        return { fila: fila, caja: caja };
    }

    function puntosCargando() {
        const f = document.createDocumentFragment();
        for (let i = 0; i < 3; i++) f.appendChild(document.createElement('span'));
        return f;
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
        burbuja(chatBody, document.createTextNode(msg), 'chat-bubble-user', 'end');
        input.value = '';
        const cargando = burbuja(chatBody, puntosCargando(), 'chat-bubble-bot typing', 'start');

        fetch('/chat-ia', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': document.querySelector('meta[name="csrf-token"]').content
            },
            body: 'mensaje=' + encodeURIComponent(msg)
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                cargando.fila.remove();
                const respuesta = burbuja(chatBody, textoConFormato(data.respuesta || 'Sin respuesta.'), 'chat-bubble-bot', 'start');
                const fuentes = (data.fuentes || []).filter(Boolean);
                if (fuentes.length) {
                    const pie = document.createElement('div');
                    pie.className = 'sp-chat-fuentes';
                    const icono = document.createElement('i');
                    icono.className = 'fas fa-database me-1';
                    pie.appendChild(icono);
                    pie.appendChild(document.createTextNode(fuentes.join(' · ')));
                    respuesta.caja.appendChild(pie);
                }
            })
            .catch(function () {
                cargando.fila.remove();
                burbuja(chatBody, document.createTextNode('No se pudo conectar con el asistente.'), 'chat-bubble-bot', 'start');
            });
    }

    document.addEventListener('DOMContentLoaded', function () {
        document.querySelectorAll('[data-chat="alternar"]').forEach(function (b) { b.addEventListener('click', toggleChat); });
        const enviar = document.getElementById('chat-enviar');
        if (enviar) enviar.addEventListener('click', sendMessage);
        const input = document.getElementById('chat-input');
        if (input) input.addEventListener('keydown', function (e) { if (e.key === 'Enter') sendMessage(); });
    });
})();
