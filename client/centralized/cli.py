import sys
import json
from client.centralized.http_client import HttpClient

def show_menu():
    print("\n=== DistIA - Menú Principal ===")
    print("1. Subir dataset")
    print("2. Crear job de entrenamiento")
    print("3. Ver estado de un job")
    print("4. Listar jobs")
    print("5. Descargar modelo")
    print("6. Actualizar lista de servidores")
    print("0. Salir")
    return input("Selecciona una opción: ").strip()

def run_cli():
    try:
        client = HttpClient()
    except Exception as e:
        print(f"Error al inicializar cliente: {e}")
        sys.exit(1)

    while True:
        option = show_menu()

        if option == "1":
            file_path = input("Ruta del dataset: ").strip()
            name = input("Nombre (opcional): ").strip() or None
            try:
                result = client.upload_dataset(file_path, name)
                print("✅ Dataset subido correctamente:")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("❌ Error:", e)

        elif option == "2":
            dataset_id = input("ID del dataset: ").strip()
            task = input("Tarea (ej: classification/regression): ").strip()
            models = input("Modelos (ej: LinearRegressor, KNNRegressor): ").strip().split(",")
            params = input("Parámetros JSON (opcional): ").strip()
            try:
                params_dict = json.loads(params) if params else {}
                result = client.create_job(dataset_id, task, models)
                print("✅ Job creado:")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("❌ Error:", e)

        elif option == "3":
            job_id = input("ID del job: ").strip()
            try:
                result = client.get_job_status(job_id)
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("❌ Error:", e)

        elif option == "4":
            user_id = input("User ID (opcional): ").strip() or None
            try:
                result = client.list_jobs(user_id)
                print(json.dumps(result, indent=2))
            except Exception as e:
                print("❌ Error:", e)

        elif option == "5":
            job_id = input("ID del job: ").strip()
            output_path = input("Ruta de salida (opcional): ").strip() or None
            try:
                path = client.download_model(job_id, output_path)
                print(f"✅ Modelo descargado en: {path}")
            except Exception as e:
                print("❌ Error:", e)

        elif option == "6":
            try:
                client.update_server_list()
            except Exception as e:
                print("❌ Error:", e)

        elif option == "0":
            print("👋 Saliendo...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")
