/* SIDPOL · ajustes globales del tema oscuro */
(function () {
    // 1) Chart.js: colores, cuadrícula y tipografía del tema (antes de que cada página cree sus gráficos)
    if (window.Chart) {
        Chart.defaults.color = '#a8b3c7';
        // Cuadrícula tenue por escala (no se cambia Chart.defaults.borderColor para no
        // desactivar la paleta automática de Chart.js en los gráficos sin colores propios)
        Chart.defaults.scale.grid.color = 'rgba(148, 163, 184, 0.12)';
        Chart.defaults.scale.border.color = 'rgba(148, 163, 184, 0.25)';
        Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
        Chart.defaults.plugins.legend.labels.usePointStyle = true;
        Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(13, 22, 40, 0.95)';
        Chart.defaults.plugins.tooltip.borderColor = 'rgba(56, 189, 248, 0.35)';
        Chart.defaults.plugins.tooltip.borderWidth = 1;
        Chart.defaults.plugins.tooltip.titleColor = '#f8fafc';
        Chart.defaults.plugins.tooltip.bodyColor = '#cbd5e1';
        Chart.defaults.plugins.tooltip.padding = 10;
        // Margen para que las etiquetas no se corten si la tipografía web carga después del cálculo
        Chart.defaults.layout.padding = { left: 6, right: 8 };
        // Ejes verticales con etiquetas de texto: ancho extra para que la etiqueta más larga no se corte
        Chart.defaults.scale.afterFit = function (scale) {
            if (!scale.isHorizontal() && scale.type === 'category') scale.width += 18;
        };
        // Cuando termina de cargar Inter, se recalculan los gráficos ya dibujados
        if (document.fonts && document.fonts.ready) {
            document.fonts.ready.then(function () {
                Object.values(Chart.instances || {}).forEach(function (c) { c.resize(); c.update('none'); });
            });
        }
    }

    document.addEventListener('DOMContentLoaded', function () {
        // 2) Marca la opción activa del menú según la URL
        var ruta = window.location.pathname.replace(/\/+$/, '') || '/';
        var alias = { '/riesgo-modalidad': '/prediccion-2026' };  // el simulador pertenece a Proyección
        ruta = alias[ruta] || ruta;
        document.querySelectorAll('.navbar .nav-link[href], .navbar .dropdown-item[href]').forEach(function (a) {
            var destino = (a.getAttribute('href') || '').replace(/\/+$/, '') || '/';
            if (destino === ruta) {
                a.classList.add('active');
                var dropdown = a.closest('.dropdown');
                if (dropdown) {
                    var toggle = dropdown.querySelector('.nav-link');
                    if (toggle) toggle.classList.add('active');
                }
            }
        });

        // 3) Contadores animados: <span data-count="6691038" data-format="compacto">0</span>
        var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        document.querySelectorAll('[data-count]').forEach(function (el) {
            var fin = parseFloat(el.getAttribute('data-count')) || 0;
            var formato = el.getAttribute('data-format');
            var sufijo = el.getAttribute('data-suffix') || '';
            var prefijo = el.getAttribute('data-prefix') || '';
            var pinta = function (v) {
                var txt;
                if (formato === 'compacto') {
                    txt = Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(2) + ' M'
                        : Math.abs(v) >= 1e3 ? Math.round(v / 1e3).toLocaleString('es-PE') + ' K'
                        : Math.round(v).toLocaleString('es-PE');
                } else {
                    txt = Math.round(v).toLocaleString('es-PE');
                }
                el.textContent = prefijo + txt + sufijo;
            };
            if (reduce) { pinta(fin); return; }
            var t0 = null, dur = 1200;
            var paso = function (t) {
                if (!t0) t0 = t;
                var p = Math.min((t - t0) / dur, 1);
                pinta(fin * (1 - Math.pow(1 - p, 3)));
                if (p < 1) requestAnimationFrame(paso);
            };
            requestAnimationFrame(paso);
        });
    });
})();
