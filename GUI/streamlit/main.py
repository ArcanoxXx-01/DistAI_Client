import streamlit as st
import pandas as pd
import time, sys

from clients.centralized.http_client import HttpClient


class ST_App:
    def __init__(self):
        try:
            self.client = HttpClient()
        except Exception as e:
            st.error(f"Error al inicializar cliente: {e}")

        self.tasks = self.client.cfg.get("tasks", ["classification", "regression"])
        self.datasets_path: str = self.client.cfg.get("datasets_data_path")
        self.trainings_path: str = self.client.cfg.get("trainings_data_path")
        self.results_path: str = self.client.cfg.get("results_data_path")

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
        with Results_page:
            self.show_results_page()
        # with Predicts_page:
        #     self.show_predicts_page()

    def show_datasets_page(self):
        st.header("Uploaded Datasets")
        self.show_table(self.datasets_path)
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
                        self.add_row(self.datasets_path, f"\n{id},{dataset_name}")
                        st.success(f"Dataset uploaded successfully! ID: {id}")
                    except Exception as e:
                        st.error(f"Error uploading dataset: {e}")
                    finally:
                        st.rerun()

    def show_trainings_page(self):
        st.header("Trainings:")
        self.show_table(self.trainings_path)
        with st.expander("Create New Training", expanded=False) as pannel:
            col1, col2 = st.columns(2)
            with col1:
                dataset_id = st.selectbox(
                    "Dataset Id", self.get_ids(self.datasets_path)
                )
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
                        row = ""
                        status = self.client.get_job_status(id).get("status", "unknown")
                        for model in selected_models:
                            row += f"\n{id},{task},{model},{status}"
                        self.add_row(self.trainings_path, row)
                    except Exception as e:
                        st.error(f"Error creating training job: {e}")
                    finally:
                        st.rerun()
        if st.button("Refresh Status"):
            try:
                status = {}
                for id in self.get_ids(self.trainings_path):
                    response = self.client.get_job_status(id)["status"]
                    status[id] = response
                df = pd.DataFrame(pd.read_csv(self.trainings_path))
                df["status"] = df["id"].map(status)
                df.to_csv(self.trainings_path, index=False)
            except Exception as e:
                st.error(f"Error updating Status: {e}")
            finally:
                st.rerun()

    def show_results_page(self):
        st.header("Training Results")

        id = st.selectbox("Training Id", self.get_ids(self.trainings_path))

        if id and st.button("Get Results"):
            with st.spinner("Fetching results..."):
                try:
                    status_response = self.client.get_job_status(id)
                    status = status_response.get("status")

                    st.write(f"Job Status: **{status}**")

                    if status == "completed":
                        # Obtener resultados
                        results = self.client.get_results(id)

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
                                    output_path = self.client.download_model(id)
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

    def show_table(self, path):
        data = pd.read_csv(path)
        df = pd.DataFrame(data)
        st.dataframe(df)

    def add_row(self, path, row: str):
        with open(path, "a") as f2:
            f2.write(f"\n{row}")
            f2.close()

    def get_ids(self, path):
        return pd.DataFrame(pd.read_csv(path))["id"].unique()
