import argparse, json
from .http_client import HttpClient


def build_parser():
    p = argparse.ArgumentParser(
        prog="client", description=" DistAI Client"
    )
    p.add_argument(
        "--server",
        required=False,
        help="URL del servidor (opcional, si se quiere sobreescribir)",
    )
    p.add_argument("--token", required=False, help="Token de autenticación (opcional)")
    sub = p.add_subparsers(dest="cmd")

    up = sub.add_parser("upload", help="Subir CSV")
    up.add_argument("csv", help="Ruta al CSV")
    up.add_argument("--name", help="Nombre del dataset")

    cj = sub.add_parser("create-job", help="Crear job de entrenamiento")
    cj.add_argument("--dataset-id", required=True)
    cj.add_argument("--task", choices=["classification", "regression"], required=True)
    cj.add_argument("--model", required=True)
    cj.add_argument("--params", default="{}", help="JSON con parámetros")

    st = sub.add_parser("status", help="Consultar estado de job")
    st.add_argument("job_id")

    ls = sub.add_parser("list", help="Listar jobs")
    ls.add_argument("--user-id", help="Filtrar por usuario (opcional)")

    dl = sub.add_parser("download", help="Descargar modelo")
    dl.add_argument("job_id")
    dl.add_argument("--output", help="Ruta de salida (opcional)")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    client = HttpClient()
    
    # if override server/token
    if getattr(args, "server", None):
        client.server = args.server.rstrip("/")
    if getattr(args, "token", None):
        client.token = args.token

    if args.cmd == "upload":
        r = client.upload_dataset(args.csv, args.name)
        print("Dataset subido:", r)
    elif args.cmd == "create-job":
        params = json.loads(args.params)
        r = client.create_job(args.dataset_id, args.task, args.model, params)
        print("Job creado:", r)
    elif args.cmd == "status":
        r = client.get_job_status(args.job_id)
        print("Status:", r)
    elif args.cmd == "list":
        r = client.list_jobs(args.user_id)
        print("Jobs:", r)
    elif args.cmd == "download":
        out = client.download_model(args.job_id, args.output)
        print("Modelo guardado en:", out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
