# ChatbotITP

<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/pdf.svg" width="100" />
<br>Asistente Virtual ITP
</h1>
<h3>◦ Interactúa con tus documentos mediante el Asistente Virtual del ITP✨</h3>
<h3>◦ Desarrollado usando &nbsp;&nbsp; <span><b>🐍Python</b></span>&nbsp;&nbsp; | &nbsp;&nbsp; <span>🦜️🔗 <b>LangChain</b></span></h3>
  <p>
    
<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style&logo=Streamlit&logoColor=white" alt="Streamlit" />
<img src="https://img.shields.io/badge/OpenAI-412991.svg?style&logo=OpenAI&logoColor=white" alt="OpenAI" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/Markdown-000000.svg?style&logo=Markdown&logoColor=white" alt="Markdown" />
  </p>
</div>

---

## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [⚙️ Features](#️-features)
- [📂 Project Structure](#-project-structure)
- [🧩 Modules](#-modules)
- [🚀 Getting Started](#-getting-started)
  - [✔️ Prerequisites](#️-prerequisites)
  - [📦 Installation](#-installation)
  - [🎮 Using the Virtual Assistant](#-using-the-virtual-assistant)
- [🤝 Contributing](#-contributing)

---

## 📍 Overview

El proyecto Asistente Virtual ITP es una aplicación Streamlit que permite a los usuarios cargar archivos PDF y DOCX e interactuar con un modelo de IA conversacional. Utiliza la API de OpenAI para interacciones conversacionales y FAISS para búsqueda de similitudes rápida. Las funcionalidades principales de este proyecto incluyen el análisis de documentos, la extracción de texto, la generación de embeddings para consultas de usuario y la provisión de respuestas relevantes basadas en el contenido del documento. Su propuesta de valor radica en simplificar el proceso de extracción de información de documentos y facilitar interacciones conversacionales con el contenido.

---

## ⚙️ Features

| Feature                | Description                           |
| ---------------------- | ------------------------------------- |
| **⚙️ Architecture**     | El sistema sigue un patrón de diseño modular, donde diferentes componentes manejan tareas como la carga de archivos, extracción de contenido de documentos, conversión de vectores, recuperación conversacional e interacción con el usuario. La aplicación utiliza la API de OpenAI, FAISS y varias utilidades para estas tareas. También incorpora una interfaz web con Streamlit.    |
| **📖 Documentation**   | La base de código proporciona documentación clara y completa, explicando el propósito y uso de cada componente y función. Incluye comentarios informativos a lo largo del código para facilitar la comprensión y el mantenimiento.    |
| **🔗 Dependencies**    | El sistema depende de bibliotecas externas como OpenAI, FAISS y Streamlit para funcionalidades clave. Estas dependencias están claramente listadas en el archivo requirements.txt del proyecto, facilitando la configuración y replicación del proyecto.    |
| **🧩 Modularity**      | La base de código está bien organizada en componentes más pequeños e intercambiables. Cada componente maneja una tarea específica, promoviendo la reutilización de código, el mantenimiento y la facilidad de prueba. El enfoque modular también permite una fácil extensión y personalización.    |
| **⚡️ Performance**      | El rendimiento del sistema está sujeto a factores externos como las respuestas de la API y los tamaños de los documentos. Sin embargo, la base de código optimiza donde es posible, utilizando FAISS para una búsqueda de similitudes eficiente y recuperación de vectores.     |
| **🔀 Version Control** | El proyecto está gestionado a través del control de versiones de Git, como se evidencia en el repositorio de GitHub. Esto permite el desarrollo colaborativo, la gestión de ramas y el seguimiento de problemas. Un historial de commits adecuado y comentarios facilitan las revisiones de código y la solución de problemas.    |
| **🔌 Integrations**    | El sistema aprovecha múltiples integraciones, principalmente con la API de OpenAI para interfaces conversacionales y FAISS para la búsqueda de similitudes. Streamlit se utiliza para proporcionar una interfaz web a los usuarios, y se podrían implementar más integraciones con servicios adicionales para mejorar la funcionalidad.    |

---

## 📂 Project Structure

Aquí puedes describir la estructura de carpetas y archivos del proyecto.

---

## 🧩 Modules

<details closed><summary>Root</summary>

| File                                                             | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---                                                              | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| [app.py](https://github.com/tu-usuario/AsistenteVirtualITP/blob/main/app.py) | Asistente Virtual ITP es una aplicación Streamlit que permite a los usuarios cargar archivos PDF y DOCX y hacer preguntas sobre el contenido. Utiliza la API de OpenAI para interacciones conversacionales, FAISS para una búsqueda de similitudes rápida, y varias utilidades para analizar y manejar documentos. Soporta funciones como analizar archivos DOCX, extraer contenido textual de PDFs y DOCX, dividir texto en fragmentos manejables, generar vectores a partir de fragmentos usando embeddings de OpenAI y FAISS, y crear una instancia de ConversationalRetrievalChain para procesar consultas de usuario. La función principal maneja la carga de archivos, entrada de usuario y muestra respuestas del bot en una interfaz Streamlit. |

</details>

---

## 🚀 Getting Started

### ✔️ Prerequisites

Antes de empezar, asegúrate de tener instalados los siguientes requisitos:

- Python 3.x
- pip

### 📦 Installation

1. **Clona el repositorio del Asistente Virtual ITP**:
```sh
git clone https://github.com/tu-usuario/AsistenteVirtualITP
```

2. **Cambia al directorio del proyecto**:
```sh
cd AsistenteVirtualITP
```

3. **Instala las dependencias**:
```sh
pip install -r requirements.txt
```

4. **Instala y configura MongoDB**:

   - Descarga MongoDB desde [aquí](https://www.mongodb.com/try/download/community) e instálalo.
   - Inicia MongoDB ejecutando:
   ```sh
   mongod
   ```

5. **Instala y configura Elasticsearch**:

   - Descarga Elasticsearch desde [aquí](https://www.elastic.co/downloads/elasticsearch) e instálalo.
   - Inicia Elasticsearch ejecutando:
   ```sh
   elasticsearch
   ```

### 🎮 Using the Virtual Assistant

Para ejecutar la aplicación:
```sh
streamlit run app.py
```

---

## 🤝 Contributing

¡Las contribuciones son siempre bienvenidas! Por favor, sigue estos pasos:

1. Haz un fork del repositorio del proyecto. Esto crea una copia del proyecto en tu cuenta que puedes modificar sin afectar el proyecto original.
2. Clona el repositorio forkeado en tu máquina local usando un cliente Git como Git o GitHub Desktop.
3. Crea una nueva rama con un nombre descriptivo (por ejemplo, `new-feature-branch` o `bugfix-issue-123`).
```sh
git checkout -b new-feature-branch
```
4. Haz cambios en la base de código del proyecto.
5. Haz commit de tus cambios en tu rama local con un mensaje de commit claro que explique los cambios que has realizado.
```sh
git commit -m 'Implementado nueva funcionalidad.'
```
6. Empuja tus cambios a tu repositorio forkeado en GitHub usando el siguiente comando:
```sh
git push origin new-feature-branch
```
7. Crea un nuevo pull request al repositorio original del proyecto. En el pull request, describe los cambios que has realizado y por qué son necesarios.

Los mantenedores del proyecto revisarán tus cambios y proporcionarán feedback o los fusionarán en la rama principal.

---
