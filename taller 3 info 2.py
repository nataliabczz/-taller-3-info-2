import os
import pydicom
import pandas as pd
import numpy as np
import cv2

class ProcesadorDICOM:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.data = [] # Aquí guardaremos los metadatos
        
        # Crear directorio de salida si no existe
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def procesar_directorio(self):
        """Escanea el directorio e identifica archivos para procesar."""
        # 4.1 Carga de archivos DICOM
        for filename in os.listdir(self.input_dir):
            filepath = os.path.join(self.input_dir, filename)
            if os.path.isfile(filepath):
                self._procesar_archivo(filepath, filename)

        # 4.3 Estructuración de los datos
        # Convertimos la lista de diccionarios a un DataFrame de Pandas
        df = pd.DataFrame(self.data)
        return df

    def _procesar_archivo(self, filepath, filename):
        """Procesa un archivo individual: lee metadatos y modifica la imagen."""
        try:
            # Intentar leer el archivo usando pydicom
            ds = pydicom.dcmread(filepath)
        except Exception as e:
            # Manejo de errores si el archivo no es un DICOM válido
            print(f"Omitiendo {filename}: No es un archivo DICOM válido. Error: {e}")
            return

        # 4.2 Extracción de metadatos (manejando tags faltantes)
        metadata = {
            "Archivo": filename,
            "PatientID": getattr(ds, 'PatientID', 'No Disponible'),
            "PatientName": str(getattr(ds, 'PatientName', 'No Disponible')),
            "StudyInstanceUID": getattr(ds, 'StudyInstanceUID', 'No Disponible'),
            "StudyDescription": getattr(ds, 'StudyDescription', 'No Disponible'),
            "StudyDate": getattr(ds, 'StudyDate', 'No Disponible'),
            "Modality": getattr(ds, 'Modality', 'No Disponible'),
            "Rows": getattr(ds, 'Rows', 'No Disponible'),
            "Columns": getattr(ds, 'Columns', 'No Disponible'),
            "Intensidad Promedio": 'No Disponible' 
        }

        # 4.4 y 4.5 Análisis y Procesamiento de Imágenes
        try:
            # Verificar si el archivo tiene datos de píxeles accesibles
            if hasattr(ds, 'pixel_array'):
                img_array = ds.pixel_array

                # 4.4 Análisis con NumPy
                # Calcular la intensidad promedio y agregarla al diccionario
                intensidad_promedio = np.mean(img_array)
                metadata["Intensidad Promedio"] = intensidad_promedio

                # 4.5 Procesamiento con OpenCV
                # 1. Normalización a 8 bits (rango 0 a 255)
                img_min = np.min(img_array)
                img_max = np.max(img_array)
                
                # Evitar división por cero si la imagen es completamente plana
                if img_max > img_min: 
                    img_norm = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_norm = img_array.astype(np.uint8)

                # 2. Ecualización del histograma
                img_eq = cv2.equalizeHist(img_norm)

                # 3. Detección de bordes con Canny
                # Usamos umbrales 50 y 150 (explicación en el README)
                edges = cv2.Canny(img_eq, 50, 150)

                # 4. Guardado de resultados
                # Usamos el StudyInstanceUID para el nombre, si no está, usamos el nombre del archivo
                base_name = metadata["StudyInstanceUID"] if metadata["StudyInstanceUID"] != 'No Disponible' else filename.replace('.dcm', '')
                
                ruta_eq = os.path.join(self.output_dir, f"{base_name}_eq.png")
                ruta_edges = os.path.join(self.output_dir, f"{base_name}_edges.png")
                
                cv2.imwrite(ruta_eq, img_eq)
                cv2.imwrite(ruta_edges, edges)

        except Exception as e:
            # Manejo del caso donde no hay imagen (ej. PR, SR) o falla el procesamiento
            print(f"El archivo {filename} no contiene imagen procesable: {e}")

        # Agregar los metadatos de este archivo a nuestra lista global
        self.data.append(metadata)

if __name__ == "__main__":
    # Define las carpetas de entrada y salida
    DIRECTORIO_ENTRADA = "./dicom_input" # Asegúrate de crear esta carpeta y poner archivos .dcm dentro
    DIRECTORIO_SALIDA = "./dicom_output"

    procesador = ProcesadorDICOM(DIRECTORIO_ENTRADA, DIRECTORIO_SALIDA)
    
    print("Iniciando procesamiento...")
    df_resultados = procesador.procesar_directorio()
    
    print("\n--- Metadatos Extraídos ---")
    print(df_resultados)
    
    # Opcional: Guardar el dataframe para revisarlo después
    df_resultados.to_csv("metadatos.csv", index=False)
    print("\nProcesamiento completado. Revisa la carpeta de salida.")