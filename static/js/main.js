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
