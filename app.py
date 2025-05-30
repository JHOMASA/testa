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

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    filename='dentalstock.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# App configuration
st.set_page_config(
    page_title="DentalStock PRO Lite",
    page_icon="🦷",
    layout="wide"
)
DATABASES_DIR = "databases"
os.makedirs(DATABASES_DIR, exist_ok=True)

# --- Database Functions ---
def get_db(tenant_id: str) -> sqlite3.Connection:
    """Get database connection with proper schema"""
    db_path = f"{DATABASES_DIR}/{tenant_id}.db"
    try:
        conn = sqlite3.connect(db_path)
        
        # Improved schema with constraints
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS productos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL CHECK(length(nombre) > 0),
            lote TEXT NOT NULL CHECK(length(lote) > 0),
            caducidad DATE NOT NULL,
            stock_actual INTEGER DEFAULT 0 CHECK(stock_actual >= 0),
            necesita_digemid BOOLEAN DEFAULT TRUE,
            codigo_digemid TEXT,
            categoria TEXT,
            precio_compra REAL NOT NULL CHECK(precio_compra >= 0),
            precio_venta REAL NOT NULL CHECK(precio_venta >= precio_compra),
            CHECK (necesita_digemid = FALSE OR codigo_digemid IS NOT NULL)
        );
        
        CREATE TABLE IF NOT EXISTS movimientos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            producto_id INTEGER NOT NULL,
            tipo TEXT NOT NULL CHECK(tipo IN ('entrada', 'salida')),
            cantidad INTEGER NOT NULL CHECK(cantidad > 0),
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (producto_id) REFERENCES productos(id) ON DELETE CASCADE
        );
        
        CREATE TABLE IF NOT EXISTS configuracion (
            tenant_id TEXT PRIMARY KEY,
            umbral_alerta_stock INTEGER DEFAULT 10 CHECK(umbral_alerta_stock > 0),
            dias_alerta_caducidad INTEGER DEFAULT 90 CHECK(dias_alerta_caducidad > 0)
        );
        
        CREATE INDEX IF NOT EXISTS idx_productos_nombre ON productos(nombre);
        CREATE INDEX IF NOT EXISTS idx_movimientos_producto ON movimientos(producto_id);
        """)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        st.error("Error de conexión con la base de datos")
        raise

# --- Helper Functions ---
def validar_digemid(codigo: str) -> bool:
    """Validate DIGEMID code format"""
    try:
        return bool(re.fullmatch(r"^DGM-[A-Z]{2,4}-\d{4}-[0-9A-Z]{3}$", codigo))
    except:
        return False

def generar_codigo_digemid(nombre: str, lote: str) -> str:
    """Generate valid DIGEMID code"""
    try:
        prefix = (nombre[:3] or "GEN").upper().strip()
        year = datetime.now().year
        suffix = (lote[:2] or "XX").upper() + "X"
        return f"DGM-{prefix}-{year}-{suffix}"
    except:
        return "DGM-GEN-0000-000"

def clasificar_producto(nombre: str, modelos: Dict[str, Any]) -> str:
    """Classify product using AI model"""
    if not nombre or not modelos.get("clasificador"):
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
        logger.warning(f"Classification failed: {str(e)}")
        return "otros"

def predecir_demanda(producto: str, tenant_id: str, modelos: Dict[str, Any]) -> str:
    """Predict demand with fallback to simple heuristic"""
    try:
        with get_db(tenant_id) as conn:
            df = pd.read_sql("""
                SELECT cantidad, fecha FROM movimientos
                JOIN productos ON movimientos.producto_id = productos.id
                WHERE productos.nombre = ?
                ORDER BY fecha DESC LIMIT 6
            """, conn, params=(producto,))
            
            # Fallback if insufficient data
            if df.empty or df['cantidad'].isnull().all():
                return "Comprar 5 unidades (sin datos históricos)"
                
            avg = df['cantidad'].mean()
            if pd.isna(avg):
                return "Comprar 5 unidades (datos inválidos)"
            
            # Use AI if available and enough data
            if len(df) >= 3 and modelos.get("predictor"):
                season = "alta" if datetime.now().month in [6,7,12] else "baja"
                prompt = f"""Predice demanda para {producto}:
                - Promedio histórico: {avg:.1f} unidades
                - Mes: {datetime.now().month} ({season})
                - Últimos movimientos: {', '.join(map(str, df['cantidad']))}
                
                Devuelve sólo: "Comprar X unidades [RAZÓN]" """
                
                output = modelos["predictor"](
                    prompt,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )
                return output[0]["generated_text"].split("\n")[0]
            
            return f"Comprar {max(5, int(avg * 1.1))} unidades (regla básica)"
            
    except Exception as e:
        logger.error(f"Demand prediction failed: {str(e)}")
        return "Comprar 5 unidades (error en predicción)"

# --- Model Loading ---
@st.cache_resource
def load_models() -> Dict[str, Any]:
    """Load AI models with proper device handling"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading models on device: {device}")
        
        # TinyLlama model
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16
        ).to(device)
        
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
            ),
            "device": device
        }
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.warning("Algunas funciones de IA estarán limitadas")
        return {"clasificador": None, "predictor": None, "device": None}

# --- Streamlit UI ---
def main():
    modelos = load_models()
    st.title("💊 DentalStock PRO - Edición Lite")
    
    # Tenant selection sidebar
    with st.sidebar:
        st.header("Configuración")
        tenant_id = st.selectbox(
            "Seleccione Farmacia",
            [f"dental_{i}" for i in range(1, 6)] + [f"clinica_{i}" for i in range(1, 6)],
            key="tenant_select"
        )
        
        if st.button("🔁 Reiniciar Base de Datos", type="secondary", help="Elimina todos los datos de esta farmacia"):
            try:
                os.remove(f"{DATABASES_DIR}/{tenant_id}.db")
                st.success("Base de datos reiniciada")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"No se pudo reiniciar: {str(e)}")
        
        # Display system info
        with st.expander("ℹ️ Información del Sistema"):
            st.write(f"**Dispositivo AI:** {modelos.get('device', 'CPU')}")
            st.write(f"**SQLite:** {sqlite3.sqlite_version}")
            st.write(f"**Modelos cargados:**")
            st.json({
                "clasificador": modelos["clasificador"] is not None,
                "predictor": modelos["predictor"] is not None
            })
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Registrar", "⚠️ Alertas", "📊 Informes", "⚙️ Configuración"])
    
    # Product Registration
    with tab1:
        st.subheader("Registro de Productos")
        with st.form("form_producto", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                nombre = st.text_input("Nombre*", max_chars=100, help="Ej: Lidocaína 2% con epinefrina")
                lote = st.text_input("Lote*", max_chars=20)
                necesita_digemid = st.checkbox("Requiere código DIGEMID", value=True)
            with col2:
                caducidad = st.date_input("Caducidad*", min_value=datetime.now().date())
                precio_compra = st.number_input("Precio Compra (S/)*", min_value=0.0, step=0.1, format="%.2f")
                precio_venta = st.number_input("Precio Venta (S/)*", min_value=0.0, step=0.1, format="%.2f")
            
            submitted = st.form_submit_button("💾 Guardar Producto")
            if submitted:
                if not nombre or not lote:
                    st.error("¡Nombre y lote son obligatorios!")
                elif precio_venta < precio_compra:
                    st.error("El precio de venta debe ser mayor al de compra")
                else:
                    try:
                        codigo_digemid = generar_codigo_digemid(nombre, lote) if necesita_digemid else None
                        if necesita_digemid and not validar_digemid(codigo_digemid):
                            st.error("¡Formato de código DIGEMID inválido!")
                        else:
                            categoria = clasificar_producto(nombre, modelos)
                            with get_db(tenant_id) as conn:
                                conn.execute("""
                                INSERT INTO productos (
                                    nombre, lote, caducidad, necesita_digemid,
                                    codigo_digemid, categoria, precio_compra, precio_venta
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    nombre.strip(), lote.strip(), caducidad, necesita_digemid,
                                    codigo_digemid, categoria, precio_compra, precio_venta
                                ))
                                conn.commit()
                                st.success(f"✅ {nombre} registrado como {categoria}")
                                st.balloons()
                    except sqlite3.IntegrityError as e:
                        st.error(f"Error en base de datos: {str(e)}")
                    except Exception as e:
                        st.error(f"Error inesperado: {str(e)}")
    
    # Alerts Tab
    with tab2:
        st.subheader("Alertas Prioritarias")
        try:
            with get_db(tenant_id) as conn:
                alertas = pd.read_sql("""
                SELECT 
                    nombre, 
                    lote, 
                    caducidad, 
                    stock_actual, 
                    codigo_digemid,
                    CASE 
                        WHEN caducidad <= date('now') THEN 'VENCIDO'
                        WHEN caducidad <= date('now', '+30 days') THEN 'PRÓXIMO A VENCER'
                        WHEN stock_actual <= 5 THEN 'STOCK CRÍTICO'
                        ELSE 'ALERTA'
                    END as estado
                FROM productos
                WHERE caducidad <= date('now', '+3 months') OR stock_actual <= 10
                ORDER BY 
                    CASE 
                        WHEN caducidad <= date('now') THEN 0
                        WHEN caducidad <= date('now', '+30 days') THEN 1
                        ELSE 2
                    END,
                    caducidad ASC
                """, conn)
                
                if not alertas.empty:
                    # Apply color coding
                    def color_alert(row):
                        if row['estado'] == 'VENCIDO':
                            return ['background-color: #ffcccc'] * len(row)
                        elif row['estado'] == 'PRÓXIMO A VENCER':
                            return ['background-color: #fff3cd'] * len(row)
                        elif row['estado'] == 'STOCK CRÍTICO':
                            return ['background-color: #f8d7da'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(
                        alertas.style.apply(color_alert, axis=1),
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "caducidad": st.column_config.DateColumn("Caducidad"),
                            "stock_actual": st.column_config.NumberColumn("Stock"),
                            "estado": st.column_config.TextColumn("Estado")
                        }
                    )
                    
                    # Download button
                    csv = alertas.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📤 Descargar Alertas (CSV)",
                        csv,
                        file_name=f"alertas_{tenant_id}_{datetime.now().date()}.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("🎉 ¡No hay alertas activas!")
        except Exception as e:
            st.error(f"Error al cargar alertas: {str(e)}")
    
    # Reports Tab
    with tab3:
        st.subheader("Informe de Gestión")
        
        # Inventory Summary
        with st.expander("📊 Resumen de Inventario"):
            try:
                with get_db(tenant_id) as conn:
                    resumen = pd.read_sql("""
                    SELECT 
                        categoria,
                        COUNT(*) as cantidad_productos,
                        SUM(stock_actual) as stock_total,
                        AVG(precio_venta) as precio_promedio
                    FROM productos
                    GROUP BY categoria
                    ORDER BY stock_total DESC
                    """, conn)
                    
                    if not resumen.empty:
                        st.dataframe(
                            resumen.style.format({
                                "precio_promedio": "S/. {:.2f}"
                            }),
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Productos", resumen['cantidad_productos'].sum())
                        col2.metric("Total Stock", resumen['stock_total'].sum())
                        col3.metric("Valor Estimado", 
                                   f"S/. {(resumen['stock_total'] * resumen['precio_promedio']).sum():,.2f}")
                    else:
                        st.warning("No hay productos registrados")
            except Exception as e:
                st.error(f"Error al generar resumen: {str(e)}")
        
        # AI Recommendations
        if st.button("🔄 Generar Recomendaciones con IA", type="primary"):
            with st.spinner("Analizando datos..."):
                try:
                    with get_db(tenant_id) as conn:
                        productos = pd.read_sql("""
                        SELECT DISTINCT nombre 
                        FROM productos
                        WHERE stock_actual <= 20 OR id IN (
                            SELECT producto_id FROM movimientos 
                            WHERE fecha >= date('now', '-3 months')
                        """, conn)
                        
                        if not productos.empty:
                            st.subheader("Recomendaciones de Compra")
                            progress_bar = st.progress(0)
                            total = len(productos)
                            recomendaciones = []
                            
                            for i, row in enumerate(productos.itertuples()):
                                progress_bar.progress((i + 1) / total)
                                rec = predecir_demanda(row.nombre, tenant_id, modelos)
                                recomendaciones.append(f"- **{row.nombre}**: {rec}")
                            
                            st.markdown("\n".join(recomendaciones))
                            st.success("Análisis completado")
                        else:
                            st.warning("No hay suficientes datos para generar recomendaciones")
                except Exception as e:
                    st.error(f"Error al generar recomendaciones: {str(e)}")
    
    # Configuration Tab
    with tab4:
        st.subheader("Configuración DIGEMID")
        
        # Configuration form
        with st.form("config_form"):
            st.info("""
            **Reglas de validación DIGEMID (Perú)**:
            - Formato: `DGM-XXX-AAAA-ABC`  
            - Ejemplo válido: `DGM-LID-2024-XY1`  
            - Solo productos médicos requieren código.
            """)
            
            umbral = st.number_input(
                "Umbral de Alerta de Stock", 
                min_value=1, 
                value=10,
                help="Número mínimo de unidades para generar alerta"
            )
            
            dias_alerta = st.number_input(
                "Días para Alerta de Caducidad",
                min_value=1,
                value=90,
                help="Días antes de caducidad para generar alerta"
            )
            
            if st.form_submit_button("💾 Guardar Configuración"):
                try:
                    with get_db(tenant_id) as conn:
                        conn.execute("""
                        INSERT OR REPLACE INTO configuracion 
                        (tenant_id, umbral_alerta_stock, dias_alerta_caducidad)
                        VALUES (?, ?, ?)
                        """, (tenant_id, umbral, dias_alerta))
                        conn.commit()
                        st.success("Configuración guardada")
                except Exception as e:
                    st.error(f"Error al guardar configuración: {str(e)}")
        
        # Example codes
        if st.checkbox("Mostrar ejemplos de códigos DIGEMID"):
            st.code("""Ejemplos válidos:
            - DGM-ANC-2024-AB1  (Anestesia)
            - DGM-LID-2025-XY2  (Lidocaína)
            - DGM-INS-2023-MK9  (Instrumental)""")

if __name__ == "__main__":
    main()
