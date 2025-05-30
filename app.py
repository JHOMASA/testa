import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import os

# Configuración de la aplicación
st.set_page_config(page_title="DentalStock PRO Lite", page_icon="🦷", layout="wide")
DATABASES_DIR = "databases"

# --- Modelos optimizados para Streamlit Gratis ---
@st.cache_resource
def load_models():
    try:
        # TinyLlama para todas las tareas (más eficiente)
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        return {
            "clasificador": pipeline(
                "text-classification",
                model="distilbert-base-uncased",  # Más ligero para clasificación
                device="cpu"
            ),
            "predictor": pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16,
                max_length=200
            )
        }
    except Exception as e:
        st.error(f"Error al cargar modelos: {str(e)}")
        return {"clasificador": None, "predictor": None}

# --- Funciones de base de datos (completas) ---
def get_db(tenant_id: str) -> sqlite3.Connection:
    os.makedirs(DATABASES_DIR, exist_ok=True)
    conn = sqlite3.connect(f"{DATABASES_DIR}/{tenant_id}.db")
    
    # Esquema completo de la base de datos
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS productos (
        id INTEGER PRIMARY KEY,
        nombre TEXT NOT NULL,
        lote TEXT,
        caducidad DATE,
        stock_actual INTEGER DEFAULT 0,
        necesita_digemid BOOLEAN DEFAULT TRUE,
        codigo_digemid TEXT CHECK (necesita_digemid = FALSE OR codigo_digemid IS NOT NULL),
        categoria TEXT,
        precio_compra REAL,
        precio_venta REAL
    );
    
    CREATE TABLE IF NOT EXISTS movimientos (
        id INTEGER PRIMARY KEY,
        producto_id INTEGER,
        tipo TEXT CHECK(tipo IN ('entrada', 'salida')),
        cantidad INTEGER,
        fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (producto_id) REFERENCES productos(id)
    );
    
    CREATE TABLE IF NOT EXISTS configuracion (
        tenant_id TEXT PRIMARY KEY,
        umbral_alerta_stock INTEGER DEFAULT 10,
        dias_alerta_caducidad INTEGER DEFAULT 90
    );
    """)
    return conn

# --- Funciones auxiliares completas ---
def validar_digemid(codigo: str) -> bool:
    return re.match(r"^DGM-[A-Z]{2,4}-\d{4}-[0-9A-Z]{3}$", codigo) is not None

def generar_codigo_digemid(nombre: str, lote: str) -> str:
    return f"DGM-{nombre[:3].upper()}-{datetime.now().year}-{lote[:2]}X"

def clasificar_producto(nombre: str, modelos: dict) -> str:
    if modelos["clasificador"] is None:
        return "otros"
    
    try:
        categorias = {
            "LABEL_0": "anestesia",
            "LABEL_1": "composite",
            "LABEL_2": "instrumental",
            "LABEL_3": "profilaxis"
        }
        result = modelos["clasificador"](f"[CLASIFICACIÓN DENTAL] {nombre}")
        return categorias.get(result[0]["label"], "otros")
    except:
        return "otros"

def predecir_demanda(producto: str, tenant_id: str, modelos: dict) -> str:
    conn = get_db(tenant_id)
    df = pd.read_sql(f"""
    SELECT fecha, cantidad 
    FROM movimientos 
    WHERE producto_id IN (SELECT id FROM productos WHERE nombre='{producto}')
    ORDER BY fecha DESC LIMIT 6
    """, conn)
    
    if len(df) < 2 or modelos["predictor"] is None:
        return f"Comprar {max(5, int(df['cantidad'].mean() * 1.1))} unidades (regla básica)"
    
    prompt = f"""Eres un experto en gestión de inventario dental. Predice la demanda para {producto} considerando:
    - Promedio histórico: {df['cantidad'].mean():.0f} unidades/mes
    - Mes actual: {datetime.now().month}
    - Temporada: {"alta" if datetime.now().month in [6,7,12] else "baja"}
    
    Respuesta breve con el formato: "Recomendación: comprar X unidades [RAZÓN]"
    """
    
    try:
        output = modelos["predictor"](
            prompt,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        return output[0]["generated_text"].split("\n")[0]
    except Exception as e:
        return f"Comprar {max(5, int(df['cantidad'].mean() * 1.1))} unidades (error: {str(e)})"

# --- Interfaz completa de Streamlit ---
def main():
    modelos = load_models()
    st.title("💊 DentalStock PRO - Edición Lite")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuración")
        tenant_id = st.selectbox(
            "Seleccione Farmacia",
            [f"dental_{i}" for i in range(1, 6)] + [f"clinica_{i}" for i in range(1, 6)]
        )
        conn = get_db(tenant_id)
        
        if st.button("🔁 Reiniciar Base de Datos"):
            os.remove(f"{DATABASES_DIR}/{tenant_id}.db")
            st.success("Base de datos reiniciada")
    
    # Pestañas principales
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Registrar", "⚠️ Alertas", "📊 Informes", "⚙️ Configuración"])
    
    with tab1:
        st.subheader("Registro de Productos")
        with st.form("form_producto", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                nombre = st.text_input("Nombre*", placeholder="Ej: Lidocaína 2% con epinefrina")
                lote = st.text_input("Lote*")
                necesita_digemid = st.checkbox("Requiere código DIGEMID", value=True)
            with col2:
                caducidad = st.date_input("Caducidad*", min_value=datetime.now().date())
                precio_compra = st.number_input("Precio Compra (S/)*", min_value=0.0, step=0.1)
                precio_venta = st.number_input("Precio Venta (S/)*", min_value=0.0, step=0.1)
            
            if st.form_submit_button("💾 Guardar Producto"):
                if not nombre or not lote:
                    st.error("¡Nombre y lote son obligatorios!")
                else:
                    codigo_digemid = generar_codigo_digemid(nombre, lote) if necesita_digemid else None
                    if necesita_digemid and not validar_digemid(codigo_digemid):
                        st.error("¡Formato de código DIGEMID inválido!")
                    else:
                        categoria = clasificar_producto(nombre, modelos)
                        conn.execute("""
                        INSERT INTO productos (nombre, lote, caducidad, necesita_digemid, codigo_digemid, categoria, precio_compra, precio_venta)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (nombre, lote, caducidad, necesita_digemid, codigo_digemid, categoria, precio_compra, precio_venta))
                        conn.commit()
                        st.success(f"✅ {nombre} registrado como {categoria}")
    
    with tab2:
        st.subheader("Alertas Prioritarias")
        alertas = pd.read_sql(f"""
        SELECT nombre, lote, caducidad, stock_actual, codigo_digemid
        FROM productos
        WHERE caducidad <= date('now', '+3 months') OR stock_actual <= 10
        ORDER BY caducidad ASC
        """, conn)
        
        if not alertas.empty:
            st.dataframe(
                alertas.style.apply(
                    lambda x: ["background: #ffcccc" if pd.to_datetime(x.caducidad) <= datetime.now() + pd.Timedelta(days=30) or x.stock_actual <= 5 else "" for _ in x],
                    axis=1
                ), hide_index=True, use_container_width=True
            )
            
            st.download_button(
                "📤 Descargar Alertas (CSV)",
                alertas.to_csv(index=False).encode('utf-8'),
                file_name=f"alertas_{tenant_id}_{datetime.now().date()}.csv",
                mime="text/csv"
            )
        else:
            st.success("🎉 ¡No hay alertas activas!")
    
    with tab3:
        st.subheader("Informe de Gestión")
        if st.button("🔄 Generar Informe con IA"):
            with st.spinner("Analizando datos..."):
                productos = pd.read_sql("SELECT nombre, stock_actual FROM productos", conn)
                
                informe = {"recomendaciones": []}
                for producto in productos["nombre"].unique():
                    recomendacion = predecir_demanda(producto, tenant_id, modelos)
                    informe["recomendaciones"].append(f"- **{producto}**: {recomendacion}")
                
                st.markdown(f"""
                ## 📑 Informe de Inventario - {tenant_id.replace('_', ' ').title()}
                **Fecha**: {datetime.now().strftime('%d/%m/%Y %H:%M')}
                
                ### 📊 Resumen General
                - Productos registrados: {len(productos)}
                - Stock promedio: {productos['stock_actual'].mean():.1f} unidades
                - Valor estimado: S/ {(productos['stock_actual'] * 150).sum():.2f} (precio referencia)
                
                ### 🛒 Recomendaciones de Compra
                {"\n".join(informe["recomendaciones"])}
                """)
    
    with tab4:
        st.subheader("Configuración DIGEMID")
        st.info("""
        **Reglas de validación DIGEMID (Perú)**:
        - Formato: `DGM-XXX-AAAA-ABC`  
        - Ejemplo válido: `DGM-LID-2024-XY1`  
        - Solo productos médicos requieren código.
        """)
        
        if st.checkbox("Mostrar datos de ejemplo"):
            st.code("""Ejemplos válidos:
            - DGM-ANC-2024-AB1
            - DGM-LID-2025-XY2
            - DGM-INS-2023-MK9""")

if __name__ == "__main__":
    main()
