import re
import os
import argparse

def modificar_xml(ruta_archivo_entrada, factor):
    # Leer el contenido del archivo XML
    with open(ruta_archivo_entrada, 'r', encoding='utf-8') as file:
        contenido = file.read()

    # Buscar todas las apariciones de 'period="exp([NUMERO])"'
    def reemplazar(match):
        numero = float(match.group(1))
        nuevo_numero = round(numero * factor, 5)
        return f'period="exp({nuevo_numero})"'

    contenido_modificado = re.sub(r'period="exp\((\d+\.?\d*)\)"', reemplazar, contenido)

    # Obtener el nombre del archivo original y crear el nuevo nombre
    directorio, nombre_archivo = os.path.split(ruta_archivo_entrada)
    nuevo_nombre_archivo = f"mod_{nombre_archivo}"
    ruta_archivo_salida = os.path.join(directorio, nuevo_nombre_archivo)

    # Escribir el contenido modificado en un nuevo archivo XML
    with open(ruta_archivo_salida, 'w', encoding='utf-8') as file:
        file.write(contenido_modificado)

    return ruta_archivo_salida

def main():
    parser = argparse.ArgumentParser(description="Modificar valores en un archivo XML.")
    parser.add_argument("ruta_archivo", type=str, help="Ruta del archivo XML a modificar.")
    parser.add_argument("factor", type=float, help="Factor por el cual se multiplicarán los números encontrados.")

    args = parser.parse_args()

    nueva_ruta = modificar_xml(args.ruta_archivo, args.factor)
    print(f"Archivo modificado guardado en: {nueva_ruta}")

if __name__ == "__main__":
    main()