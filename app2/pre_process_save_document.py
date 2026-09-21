import json 
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)
logger.addHandler(logging.StreamHandler())

def make_json_serializable(obj):

    """
    - 객체를 JSON 직렬화 가능한 타입으로 재귀적으로 변합니다.
    """

    if isinstance(obj, dict):
        return {
            k: make_json_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [
            make_json_serializable(item)
            for item in obj 
        ]
    else:
        try:
            json.dumps(obj)
            return obj 
        except TypeError:
            return str(obj)

def clean_metadata(metadata):

    """
    - Pinecone 오류를 방지하기 위해 메타이데이터에서 null 값을 제거함
    """

    if not metadata:
        return {}

    cleaned = {}

    for k, v in metadata.items():
        if v is None:
            continue

        elif isinstance(v, dict):
            nested_clean = clean_metadata(v)

            if nested_clean:
                cleaned[k] = nested_clean

        elif isinstance(v, list):
            cleaned_list = []

            for i in v:
                if isinstance(i, dict):
                    cleaned_item = clean_metadata(i)

                    if cleaned_item:
                        cleaned_list.append(cleaned_item)

                    else:
                        cleaned_list.append(i)

                cleaned[k] = cleaned_list

        else:
            cleaned[k] = v 

    return cleaned 

def document_loading():
    """PDF 문서를 로드하고 처리하여 벡터 저장소에 추가함"""

    import os
    from pathlib import Path 

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore
    from models import get_embeddings

    embed_model = get_embeddings()
    host = os.getenv("PINECONE_HOST")

    if not host:
        logger.error("PINECONE_HOST enviroment variable is not set")
        return []

    pc = Pinecone()
    index = pc.Index(host = host)
    pinecone_vctorstore = PineconeVectorStore(
        index = index,
        embedding = embed_model 
    )

    project_root = Path(__file__).parent.parent 
    pdf_paths = [project_root.joinpath("data", "2403.05530v5.pdf")]

    for path in pdf_paths:
        output_filename = path.stem + ".json"
        output_path = project_root.joinpath("data", output_filename)

        logger.info(f"{path} 처리 중...")
        import pymupdf 
        import pymupdf4llm

        pdf_document: pymupdf.Document = pymupdf.open(path)
        pdf_2_json = pymupdf4llm.to_markdown(
            pdf_document,
            dpi = 300,
            page_width = 2480,
            page_chunks = True 
        )

        with open(output_path, "w", encoding="utf-8") as f:
            serializable_data = make_json_serializable(pdf_2_json)
            json.dump(serializable_data, f, ensure_ascii=False)

        logger.info(f"JSON 데이터가 {output_path}에 저장되었습니다.")

        all_docs = []

        if isinstance(pdf_2_json, list):
            for p in pdf_2_json:
                if isinstance(p, dict):
                    clean_meta = clean_metadata(p.get("metadata", {}))

                    all_docs.append(
                        Document(
                            page_content = p.get("text", ""),
                            metadata = clean_meta,
                        )
                    )
                else:
                    logger.warning(
                        f"예상치 못한 페이지 형식 발견(딕셔너리 아님): {type(p)}"
                    )
        else:
            logger.warning(f"pdf_2_json 형식이 리스트가 아님: {type(pdf_2_json)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )

        splited_docs = text_splitter.split_documents(all_docs)
        pinecone_vctorstore.add_documents(splited_docs)
        logger.info(
            f"{len(splited_docs)}개의 문서를 Pinecone 벡터 저장소에 추가했습니다."
        )

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    document_loading()