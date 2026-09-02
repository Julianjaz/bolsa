from api.routes import analyze_symbol, AnalyzeRequest
import json
import logging

logging.basicConfig(level=logging.INFO)

def run_test():
    # Creamos un request de prueba, con fecha de análisis en el pasado para reproducibilidad
    req = AnalyzeRequest(
        symbol="NVDA",
        timeframe="1D",
        analysis_date="2024-05-15",
        use_gemini=False # Sin Gemini para que la prueba sea rápida y no requiera API KEY configurada aquí
    )
    
    print("Enviando petición simulada a POST /analyze...")
    print(f"Request: {req}")
    
    try:
        response = analyze_symbol(req)
        print("\n--- Respuesta del Motor de Decisión ---")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"Error durante el test: {e}")

if __name__ == "__main__":
    run_test()
