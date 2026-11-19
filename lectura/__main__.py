"""Interfaz de consola para el método de evaluación de lectura oral."""

import argparse
import os
from itertools import islice
from pathlib import Path

from .config import CFG, OUTPUT_ROOT, REFERENCE_PATH, ROOT
from .entradas import R2, locales


def main():
    parser = argparse.ArgumentParser(description="Evaluación de lectura oral en español")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("modelos", help="Descargar y cargar los seis modelos fijados")
    prep.add_argument("--check", action="store_true", help="Comprobar la carga sin acceder a Hugging Face")
    run = commands.add_parser("evaluar", help="Exportar un Excel por grabación")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--audio", type=Path, help="Una grabación")
    source.add_argument("--carpeta", type=Path, help="Carpeta local (por defecto: audios/)")
    source.add_argument("--r2", action="store_true", help="Leer los audios de Cloudflare R2")
    run.add_argument("--prefijo", default="audios/", help="Prefijo de entrada R2")
    run.add_argument("--publicar-en", metavar="PREFIJO", help="Subir también cada Excel a R2")
    run.add_argument("--pasaje", type=Path, default=REFERENCE_PATH, help="Pasaje UTF-8 sin encabezados")
    run.add_argument("--control", type=Path, help="Texto de control privado para añadir únicamente WER al resumen de --audio")
    run.add_argument("--salida", type=Path, default=OUTPUT_ROOT)
    run.add_argument("--limite", type=int, help="Evaluar solo las primeras N grabaciones")
    run.add_argument("--sin-llm", action="store_true", help="Evaluar solo con reglas locales (ablación)")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    if args.command == "modelos":
        from .preparacion import preparar
        preparar(args.check)
        return

    if args.limite is not None and args.limite < 1:
        parser.error("--limite debe ser positivo")
    if args.publicar_en is not None and (not args.r2 or not args.publicar_en.strip("/")):
        parser.error("--publicar-en requiere --r2 y un prefijo de destino")
    if args.control and not args.audio:
        parser.error("--control se aplica a una sola grabación y requiere --audio")
    if not args.sin_llm and not os.environ.get("OPENAI_API_KEY"):
        parser.error("Para ejecutar el método completo se requiere OPENAI_API_KEY en el entorno o en .env")
    from .pipeline import evaluar_audio
    from .modelos import ModelPool
    cfg = dict(CFG, use_llm=not args.sin_llm)
    pool = ModelPool(cfg)

    def evaluate(path):
        result = evaluar_audio(path, args.pasaje, cfg, pool, args.salida, control_path=args.control)
        return Path(result["excel_path"])

    count = 0
    try:
        if args.r2:
            remote = R2()
            for key in remote.claves(args.prefijo, args.limite):
                with remote.audio(key, args.salida / "_r2") as audio:
                    excel = evaluate(audio)
                if args.publicar_en:
                    remote.publicar(excel, args.salida, args.publicar_en)
                count += 1
        else:
            paths = [args.audio] if args.audio else locales(args.carpeta or ROOT / "audios")
            for audio in islice(paths, args.limite):
                evaluate(audio)
                count += 1
        if not count:
            parser.error("No se encontraron grabaciones. Revisar la carpeta, el formato o el prefijo R2.")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
