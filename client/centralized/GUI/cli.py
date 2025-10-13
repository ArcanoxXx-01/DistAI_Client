from curses.ascii import isdigit
import sys
import json
from client.centralized.http_client import HttpClient
from threading import Thread


class ClientGUI:
    def __init__(self):
        self.regression_models = []
        self.classification_models = []
        self.datasets_id = []
        self.tasks = ["regression", "classification"]
        self.jobs_id = []
        self.results = {}
        try:
            self.client = HttpClient()
        except Exception as e:
            print(f"Error al inicializar cliente: {e}")
            sys.exit(1)
        # self.run()

    def _show_menu(self):
        print("\n=== DistIA - Menú Principal ===")
        print("1. Subir dataset")
        print("2. Crear entrenamiento")
        print("3. Ver estado de un entrenamiento")
        print("4. Listar entrenamientos")
        # print("5. Descargar modelo")
        # print("6. Actualizar lista de servidores")
        print("0. Salir")
        return input("Selecciona una opción: ").strip()

    def run(self):
        while True:
            option = self._show_menu()

            # ======== Upload DataSet ===========
            if option == "1":
                file_path = input("Ruta del dataset: ").strip()
                try:
                    response = self.client.upload_dataset(file_path)
                    print("✅ Dataset subido correctamente:")
                    id = response.get("dataset_id")
                    print(f"Id del dataset: {id}")
                    self.datasets_id.append(id)
                except Exception as e:
                    print("❌ Error:", e)

            # ======== Create Training ==========
            elif option == "2":
                dataset_id = self._get_dataset_id()
                task = self._get_task()
                models = self._get_models()
                try:
                    response = self.client.create_job(dataset_id, task, models)
                    print("✅ Entrenamiento creado:")
                    id = response.get("training_id")
                    print(f"Id del entrenamiento: {id}")
                    self.jobs_id.append(id)
                except Exception as e:
                    print("❌ Error:", e)

            # ====== Get Status + Metrics ========
            elif option == "3":
                job_id = self._get_job_id()
                try:
                    # status = self.client.get_job_status(job_id)
                    # self._print_status(status)
                    results = self.client.get_results(job_id)
                    self._print_all_results(results)
                except Exception as e:
                    print("❌ Error:", e)

            # elif option == "4":
            #     try:
                    # pass
            #     except Exception as e:
            #         print("❌ Error:", e)

            # elif option == "5":
            #     job_id = input("ID del job: ").strip()
            #     output_path = input("Ruta de salida (opcional): ").strip() or None
            #     try:
            #         path = client.download_model(job_id, output_path)
            #         print(f"✅ Modelo descargado en: {path}")
            #     except Exception as e:
            #         print("❌ Error:", e)

            # elif option == "6":
            #     try:
            #         client.update_server_list()
            #     except Exception as e:
            #         print("❌ Error:", e)

            elif option == "0":
                print("👋 Saliendo...")
                break
            else:
                print("Opción no válida. Intenta de nuevo.")

    def _print_list_option(self, li: list):
        max_len = 0
        for i in li:
            max_len = max(max_len, len(i))
        max_len = min(max_len, 10)

        print(f"|{" "*(max_len+5)}| opt |")

        for i in range(len(li)):
            row = "| "
            l = len(li[i])
            if l > max_len:
                row += f"{li[i][0:max_len]}... |  {i}  |"
            else:
                row += f"{li[i]}{" "*(max_len-l+3)} |  {i}  |"
            print(row)

    def _insNumber(self, s: str) -> bool:
        res = True
        for c in s:
            res = res and isdigit(c)
        return res

    def _get_dataset_id(self):
        self._print_list_option(self.datasets_id)
        dataset_index = 0
        print("Selecciona un dataset (0 por defecto)")
        while True:
            dataset_index = input().strip()
            if dataset_index == "":
                dataset_index = 0
                break
            if self._insNumber(dataset_index):
                dataset_index = int(dataset_index)
                if dataset_index >= len(self.datasets_id):
                    print(
                        f"Seleccione una opcion válida (entre 0 y {len(self.datasets_id)-1})"
                    )
                else:
                    break
        return self.datasets_id[dataset_index]

    def _get_task(self):
        self._print_list_option(self.tasks)
        print("Seleccione una de las opciones:")
        while True:
            task = input().strip()
            if task == "0" or task == "1":
                break
            print("Selecionó una opción inválida, vuelva a intentarlo")
        return self.tasks[int(task)]

    def _get_models(self, task):
        models = (
            self.regression_models
            if task == "regression"
            else self.classification_models
        )
        self._print_list_option(models)

        print("\nSeleccione los modelos a entrenar (separados por espacios)\n")

        while True:
            indexs = input().strip().split()
            flag = False

            for i in range(len(indexs)):
                if not self._insNumber(indexs[i]):
                    flag = True
                else:
                    if int(indexs[i]) > len(models) - 1:
                        flag = True
                if flag:
                    print("Selecionó una opción inválida, vuelva a intentarlo")
                    indexs = []
                    break

                indexs[i] = int(indexs[i])

            if len(indexs) > 0:
                return [models[ind] for ind in indexs]

    def _get_job_id(self):
        self._print_list_option(self.jobs_id)
        job_index = 0
        print("Selecciona una opcion (0 por defecto)")
        while True:
            job_index = input().strip()
            if job_index == "":
                job_index = 0
                break
            if self._insNumber(job_index):
                job_index = int(job_index)
                if job_index >= len(self.jobs_id):
                    print(
                        f"Seleccione una opcion válida (entre 0 y {len(self.jobs_id)-1})"
                    )
                else:
                    break
        return self.jobs_id[job_index]

    def _print_results_single_model(self, model: dict):
        print(f"Modelo: {model.get("model_name", "Nombre Desconocido")}\n")
        self._print_status(model.get("status", "unknown"))
        print("Métricas del modelo:\n", model.get("metrics", {}), "\n")
        print(f"Errores {model.get("error","-")}\n")

    def _print_all_results(self, results: dict):

        id = results.get("training_id")
        if len(id) > 10:
            id = id[0:10] + "..."

        print(f"Resultados del entrenamiento {id} :\n")
        print(f"Dataset id: {results.get("dataset_id"," ")[0:10]}...\n")
        print(f"Tipo de entrenamiento: {results.get("train_type","~")}\n")
        self._print_status(results.get("status", "unknown"))
        print(f"Created at: {results.get("created_at")}")
        print(f"Started at: {results.get("started_at", None)}")
        print(f"Completed at: {results.get("completed_at", None)}")
        print("\nResultados y métricas de cada modelo:\n")

        for model in results.get("results", []):
            self._print_results_single_model(model)

    def _print_status(self, status):
        s = ""
        if status == "completed":
            s = "✅ Completado"
        elif status == "running":
            s = "⏩ Corriendo"
        elif status == "pending":
            s = "⏸️ En cola o pausado"
        elif status == "failed":
            s = "❌ Fallido ❌"
        else:
            s = "⚠️ Estado desconocido ⚠️"

        print(f"Estado actual: {s}\n")
