import streamlit as st
import pandas as pd
import time, sys

from clients.centralized.http_client import HttpClient


class ST_App:
    def __init__(self):
        try:
            self.client = HttpClient()
        except Exception as e:
            print(f"Error al inicializar cliente: {e}")
            sys.exit(1)

        self.tasks = self.client.cfg.get("tasks", ["classification", "regression"])
        self.datasets_table_path: str = self.client.cfg.get("datasets_data_path")
        self.trainings_table_path: str = self.client.cfg.get("trainings_data_path")
        self.init_app()

    def init_app(self):
        st.set_page_config(
            page_title=self.client.cfg.get("TITTLE", "AI Training Platform"),
            page_icon=self.client.cfg.get("ICON", "🤖"),
            layout=self.client.cfg.get("LAYOUT", "wide"),
            initial_sidebar_state=self.client.cfg.get("SIDEBAR", "expanded"),
        )
        st.title(self.client.cfg.get("TITTLE", "AI Training Platform"))
        self.show_content_app()

    def show_content_app(self):
        Dataset_page, Trainings_page, Results_page, Predicts_page = st.tabs(
            ["Datasets", "Trainings", "Results", "Predicts"]
        )
        with Dataset_page:
            self.show_datasets_page()
        with Trainings_page:
            self.show_trainings_page()
        # with Results_page:
        #     self.show_results_page()
        # with Predicts_page:
        #     self.show_predicts_page()

    def show_datasets_page(self):

        st.header("Uploaded Datasets")
        self.show_table(self.datasets_table_path)

        with st.expander("Upload New Dataset", expanded=False):
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            dataset_name = st.text_input("Dataset Name", value="My Dataset")

            if (
                st.button("Upload Dataset")
                and uploaded_file is not None
                and dataset_name != ""
            ):
                with st.spinner("Uploading dataset..."):
                    try:
                        temp_path = f"data/tmp/{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                            f.close()

                        response = self.client.upload_dataset(temp_path)
                        id = response.get("dataset_id")

                        self.add_row(self.datasets_table_path, f"\n{id},{dataset_name}")

                        st.success(f"Dataset uploaded successfully! ID: {id}")

                    except Exception as e:
                        st.error(f"Error uploading dataset: {e}")
                    finally:
                        st.rerun()

    def show_trainings_page(self):
        st.header("Training Jobs")
        self.show_table(self.trainings_table_path)

        with st.expander("Create New Training Job", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                dataset_id = st.text_input("Dataset ID")
                task = st.selectbox("Task Type", self.tasks)

            with col2:
                try:
                    models_response = self.client.get_models(task)
                    available_models = models_response.get("models", [])
                    selected_models = st.multiselect(
                        "Select Models",
                        available_models,
                        default=available_models[:2] if available_models else [],
                    )
                except Exception as e:
                    st.error(f"Error loading models: {e}")
                    selected_models = []

                # seed = st.number_input("Random Seed", value=42)

            if st.button("Start Training") and dataset_id and selected_models and task:
                with st.spinner("Creating training job..."):
                    try:
                        response = self.client.create_job(
                            dataset_id=dataset_id,
                            task=task,
                            models=selected_models,
                        )

                        id = response.get("training_id")
                        st.success(f"Training job created! Job ID: {id}")

                    except Exception as e:
                        st.error(f"Error creating training job: {e}")

        # Lista de trabajos
        st.subheader("Training Jobs")
        if st.button("Refresh Jobs"):
            self.jobs_id = self.load_jobs()

        if self.jobs_id:
            jobs_df = pd.DataFrame(self.jobs_id)
            st.dataframe(jobs_df)
        else:
            st.info("No training jobs found")

    def show_results_page(self):
        st.header("Training Results")

        # Seleccionar job para ver resultados
        job_id = st.text_input("Enter Job ID to view results")

        if job_id and st.button("Get Results"):
            with st.spinner("Fetching results..."):
                try:
                    # Primero verificar estado
                    status_response = self.client.get_job_status(job_id)
                    status = status_response.get("status")

                    st.write(f"Job Status: **{status}**")

                    if status == "completed":
                        # Obtener resultados
                        results = self.client.get_results(job_id)

                        # Mostrar resultados
                        if results:
                            st.subheader("Training Results")

                            # Mostrar métricas por modelo
                            for model_name, metrics in results.items():
                                with st.expander(f"Model: {model_name}"):
                                    if isinstance(metrics, dict):
                                        for metric, value in metrics.items():
                                            st.write(f"{metric}: {value:.4f}")
                                    else:
                                        st.write(metrics)

                            # Botón para descargar modelo
                            if st.button("Download Best Model"):
                                try:
                                    # Asumiendo que el primer modelo es el mejor por ahora
                                    best_model = list(results.keys())[0]
                                    output_path = self.client.download_model(job_id)
                                    st.success(f"Model downloaded to: {output_path}")
                                except Exception as e:
                                    st.error(f"Error downloading model: {e}")

                    elif status == "failed":
                        st.error("Job failed. Please check the server logs.")
                    else:
                        st.info(f"Job is still running. Current status: {status}")

                except Exception as e:
                    st.error(f"Error fetching results: {e}")

    def show_predicts_page(self):
        st.header("Make Predictions")

        col1, col2 = st.columns(2)

        with col1:
            job_id = st.text_input("Job ID for Model")
            model_name = st.text_input("Model Name")

        with col2:
            uploaded_pred_file = st.file_uploader(
                "Upload prediction dataset", type="csv", key="pred_file"
            )

        if (
            st.button("Generate Predictions")
            and job_id
            and model_name
            and uploaded_pred_file
        ):
            with st.spinner("Generating predictions..."):
                try:
                    # Guardar archivo temporalmente
                    temp_path = f"/tmp/pred_{uploaded_pred_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_pred_file.getvalue())

                    # Hacer predicción
                    predictions = self.client.predict(job_id, model_name, temp_path)

                    # Mostrar resultados de predicción
                    st.subheader("Prediction Results")

                    if isinstance(predictions, dict):
                        # Si son métricas o resumen
                        for key, value in predictions.items():
                            st.write(f"{key}: {value}")
                    else:
                        # Si son las predicciones directamente
                        st.write(predictions)

                    # Opción para descargar predicciones
                    if st.button("Download Predictions"):
                        # Implementar descarga de predicciones
                        pass

                except Exception as e:
                    st.error(f"Error generating predictions: {e}")

    def upload_dataset(self, file_path: str, name: str = None):
        return self.client.upload_dataset(file_path, name)

    def create_job(
        self,
        dataset_id: str,
        task: str,
        models: list = [],
        params=None,
        train_test_split: float = 0.2,
        seed: int = 42,
    ):
        return self.client.create_job(
            dataset_id, task, models, params, train_test_split, seed
        )

    def get_job_status(self, job_id: str):
        return self.client.get_job_status(job_id)

    def list_jobs(self, user_id: str = None):
        return self.client.list_jobs(user_id)

    def download_model(self, job_id: str, output_path: str = None) -> str:
        return self.client.download_model(job_id, output_path)

    def predict(self, job_id: str, model_name: str, dataset_path: str):
        return self.client.predict(job_id, model_name, dataset_path)

    def load_datasets(self):
        """Cargar datasets desde el servidor"""
        try:
            # Nota: Necesitarías agregar un endpoint para listar datasets en tu API
            # Por ahora, asumiremos que tenemos una lista local o usaremos otra estrategia
            return []
        except Exception as e:
            st.error(f"Error cargando datasets: {e}")
            return []

    def load_jobs(self):
        """Cargar trabajos desde el servidor"""
        try:
            response = self.client.list_jobs()
            return response.get("jobs", [])
        except Exception as e:
            st.error(f"Error cargando trabajos: {e}")
            return []

    def show_table(self, path):
        data = pd.read_csv(path)
        df = pd.DataFrame(data)
        st.dataframe(df)

    def add_row(self, path , row: str):
        with open(path, "a") as f:
            f.wrtie(f"\n{row}")
            f.close()
            
            
