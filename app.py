import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(filename='dentalstock.log', level=logging.INFO)

# App configuration
st.set_page_config(page_title="DentalStock PRO Lite", page_icon="🦷", layout="wide")
DATABASES_DIR = "databases"

# --- Optimized Models for Free Tier ---
@st.cache_resource
def load_models():
    try:
        # Simplified model loading without device_map
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16
        )
        
        # Move model to GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        return {
            "clasificador": pipeline(
                "text-classification",
                model="distilbert-base-uncased",
                device=device
            ),
            "predictor": pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
                torch_dtype=torch.float16,
                max_length=200
            )
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {"clasificador": None, "predictor": None}
Fix 3: Robust Demand Prediction
Update your predecir_demanda function:

python
def predecir_demanda(producto: str, tenant_id: str, modelos: dict) -> str:
    try:
        conn = get_db(tenant_id)
        df = pd.read_sql("""
            SELECT cantidad 
            FROM movimientos 
            WHERE producto_id IN (SELECT id FROM productos WHERE nombre=?)
            ORDER BY fecha DESC LIMIT 6
        """, conn, params=(producto,))
        
        # Handle empty/invalid data
        if df.empty or df['cantidad'].isnull().all():
            return "Comprar 5 unidades (sin datos históricos)"
            
        avg = df['cantidad'].mean()
        if pd.isna(avg):
            return "Comprar 5 unidades (datos inválidos)"
            
        if len(df) < 2 or modelos["predictor"] is None:
            return f"Comprar {max(5, int(avg * 1.1))} unidades (regla básica)"
        
        # AI prediction with error handling
        try:
            prompt = f"""Predice demanda para {producto}:
            - Promedio: {avg:.0f} unidades/mes
            - Mes: {datetime.now().month}
            - Temporada: {"alta" if datetime.now().month in [6,7,12] else "baja"}
            
            Respuesta: "Comprar X unidades [RAZÓN]"
            """
            
            output = modelos["predictor"](
                prompt,
                max_new_tokens=100,
                temperature=0.7
            )
            return output[0]["generated_text"].split("\n")[0]
        except:
            return f"Comprar {max(5, int(avg * 1.1))} unidades (fallback)"
            
    except Exception as e:
        return "Comprar 5 unidades (error en predicción)"

# --- Database Functions ---
def get_db(tenant_id: str) -> sqlite3.Connection:
    os.makedirs(DATABASES_DIR, exist_ok=True)
    db_path = f"{DATABASES_DIR}/{tenant_id}.db"
    try:
        conn = sqlite3.connect(db_path)
        
        # Verify SQLite version
        min_version = (3, 35, 0)
        current_version = tuple(map(int, sqlite3.sqlite_version.split(".")))
        if current_version < min_version:
            st.warning(f"SQLite {'.'.join(map(str, min_version))}+ recommended (found {sqlite3.sqlite_version})")
        
        # Database schema
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
    except Exception as e:
        logging.error(f"Database error ({db_path}): {str(e)}")
        st.error("Failed to connect to database")
        raise

# --- Helper Functions ---
def validar_digemid(codigo: str) -> bool:
    try:
        return re.match(r"^DGM-[A-Z]{2,4}-\d{4}-[0-9A-Z]{3}$", codigo) is not None
    except:
        return False

def generar_codigo_digemid(nombre: str, lote: str) -> str:
    try:
        return f"DGM-{nombre[:3].upper()}-{datetime.now().year}-{lote[:2]}X"
    except:
        return "DGM-XXX-0000-000"

def clasificar_producto(nombre: str, modelos: dict) -> str:
    if not nombre or modelos["clasificador"] is None:
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
    except Exception as e:
        logging.warning(f"Classification failed for '{nombre}': {str(e)}")
        return "otros"

def predecir_demanda(producto: str, tenant_id: str, modelos: dict) -> str:
    try:
        conn = get_db(tenant_id)
        df = pd.read_sql("""
            SELECT fecha, cantidad 
            FROM movimientos 
            WHERE producto_id IN (SELECT id FROM productos WHERE nombre=?)
            ORDER BY fecha DESC LIMIT 6
        """, conn, params=(producto,))
        
        # Handle empty/invalid data
        if df.empty or df['cantidad'].isnull().all():
            return f"Comprar 5 unidades (sin datos históricos)"
            
        avg = df['cantidad'].mean()
        if pd.isna(avg):
            return f"Comprar 5 unidades (datos inválidos)"
            
        if len(df) < 2 or modelos["predictor"] is None:
            return f"Comprar {max(5, int(avg * 1.1))} unidades (regla básica)"
        
        # AI prediction
        prompt = f"""Eres un experto en gestión de inventario dental. Predice la demanda para {producto} considerando:
        - Promedio histórico: {avg:.0f} unidades/mes
        - Mes actual: {datetime.now().month}
        - Temporada: {"alta" if datetime.now().month in [6,7,12] else "baja"}
        
        Respuesta breve con el formato: "Recomendación: comprar X unidades [RAZÓN]"
        """
        
        output = modelos["predictor"](
            prompt,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        return output[0]["generated_text"].split("\n")[0]
        
    except Exception as e:
        logging.error(f"Demand prediction failed: {str(e)}")
        return f"Comprar 5 unidades (error)"

# --- Streamlit UI ---
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
        
        if st.button("🔁 Reiniciar Base de Datos"):
            try:
                os.remove(f"{DATABASES_DIR}/{tenant_id}.db")
                st.success("Base de datos reiniciada")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Main tabs
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
                    try:
                        codigo_digemid = generar_codigo_digemid(nombre, lote) if necesita_digemid else None
                        if necesita_digemid and not validar_digemid(codigo_digemid):
                            st.error("¡Formato de código DIGEMID inválido!")
                        else:
                            categoria = clasificar_producto(nombre, modelos)
                            conn = get_db(tenant_id)
                            conn.execute("""
                            INSERT INTO productos (
                                nombre, lote, caducidad, necesita_digemid, 
                                codigo_digemid, categoria, precio_compra, precio_venta
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                nombre, lote, caducidad, necesita_digemid, 
                                codigo_digemid, categoria, precio_compra, precio_venta
                            ))
                            conn.commit()
                            st.success(f"✅ {nombre} registrado como {categoria}")
                    except Exception as e:
                        st.error(f"Error al guardar: {str(e)}")
    
    with tab2:
        st.subheader("Alertas Prioritarias")
        try:
            conn = get_db(tenant_id)
            alertas = pd.read_sql("""
            SELECT nombre, lote, caducidad, stock_actual, codigo_digemid
            FROM productos
            WHERE caducidad <= date('now', '+3 months') OR stock_actual <= 10
            ORDER BY caducidad ASC
            """, conn)
            
            if not alertas.empty:
                st.dataframe(
                    alertas.style.apply(
                        lambda x: ["background: #ffcccc" 
                                 if pd.to_datetime(x.caducidad) <= datetime.now() + pd.Timedelta(days=30) 
                                 or x.stock_actual <= 5 
                                 else "" for _ in x],
                        axis=1
                    ), 
                    hide_index=True, 
                    use_container_width=True
                )
                
                st.download_button(
                    "📤 Descargar Alertas (CSV)",
                    alertas.to_csv(index=False).encode('utf-8'),
                    file_name=f"alertas_{tenant_id}_{datetime.now().date()}.csv",
                    mime="text/csv"
                )
            else:
                st.success("🎉 ¡No hay alertas activas!")
        except Exception as e:
            st.error(f"Error al cargar alertas: {str(e)}")
    
    with tab3:
        st.subheader("Informe de Gestión")
        if st.button("🔄 Generar Informe con IA"):
            with st.spinner("Analizando datos..."):
                try:
                    conn = get_db(tenant_id)
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
                except Exception as e:
                    st.error(f"Error al generar informe: {str(e)}")
    
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
