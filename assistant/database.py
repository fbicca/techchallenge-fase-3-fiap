"""
Base de dados estruturada (SQLite) simulando prontuário / registros clínicos.

Uso acadêmico: consultas SQL e contextualização da LLM com dados do paciente.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "db" / "prontuario_demo.sqlite"


def get_default_db_path() -> Path:
    return DEFAULT_DB_PATH


def ensure_database(db_path: Path | None = None) -> Path:
    """Cria o arquivo SQLite e as tabelas com dados de exemplo, se não existir."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS pacientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            data_nascimento TEXT,
            sexo TEXT,
            alergias TEXT,
            observacoes TEXT
        );

        CREATE TABLE IF NOT EXISTS registros_clinicos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paciente_id INTEGER NOT NULL,
            data_registro TEXT NOT NULL,
            tipo TEXT NOT NULL,
            resumo TEXT NOT NULL,
            FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
        );

        CREATE TABLE IF NOT EXISTS medicacoes_ativas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paciente_id INTEGER NOT NULL,
            medicamento TEXT NOT NULL,
            dose TEXT,
            inicio TEXT,
            FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
        );
        """
    )

    # Dados fictícios para demonstração (não são pacientes reais)
    cur.executemany(
        "INSERT INTO pacientes (nome, data_nascimento, sexo, alergias, observacoes) VALUES (?,?,?,?,?)",
        [
            (
                "Paciente Demo A",
                "1985-03-12",
                "F",
                "Penicilina",
                "Hipertensa, em acompanhamento ambulatorial.",
            ),
            (
                "Paciente Demo B",
                "1972-11-01",
                "M",
                "Nenhuma conhecida",
                "Diabetes tipo 2; última HbA1c 7,1% (laboratório fictício).",
            ),
            (
                "Paciente Demo C",
                "1999-07-22",
                "F",
                "Látex",
                "Gestante, 24ª semana (cenário ilustrativo).",
            ),
        ],
    )
    cur.executemany(
        """INSERT INTO registros_clinicos (paciente_id, data_registro, tipo, resumo) VALUES (?,?,?,?)""",
        [
            (1, "2025-01-10", "consulta", "PA 130x85; queixa de cefaleia leve; orientações gerais."),
            (1, "2025-02-05", "exame", "Hemograma sem alterações significativas (referência fictícia)."),
            (2, "2025-01-20", "consulta", "Revisão de medicação antidiabética; orientação dietética."),
            (2, "2025-03-01", "laboratorio", "Glicemia de jejum 118 mg/dL (ilustrativo)."),
            (3, "2025-02-15", "pré-natal", "US obstétrica sem intercorrências relatadas (ilustrativo)."),
        ],
    )
    cur.executemany(
        """INSERT INTO medicacoes_ativas (paciente_id, medicamento, dose, inicio) VALUES (?,?,?,?)""",
        [
            (1, "Losartana", "50 mg 1x/dia", "2024-06-01"),
            (2, "Metformina", "850 mg 2x/dia", "2023-09-15"),
            (3, "Ácido fólico", "5 mg 1x/dia", "2025-01-01"),
        ],
    )
    conn.commit()
    conn.close()
    return path


def make_sqlalchemy_engine(db_path: Path | None = None) -> Engine:
    path = ensure_database(db_path)
    return create_engine(f"sqlite:///{path}", future=True)


def fetch_patient_context_text(patient_id: int, db_path: Path | None = None) -> str:
    """
    Monta um resumo textual dos dados estruturados do paciente para injetar no prompt da LLM.
    """
    path = ensure_database(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM pacientes WHERE id = ?", (patient_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return f"(Nenhum paciente encontrado com id={patient_id}.)"

    p = dict(row)
    cur.execute(
        "SELECT data_registro, tipo, resumo FROM registros_clinicos WHERE paciente_id = ? ORDER BY data_registro DESC",
        (patient_id,),
    )
    regs = [dict(r) for r in cur.fetchall()]
    cur.execute(
        "SELECT medicamento, dose, inicio FROM medicacoes_ativas WHERE paciente_id = ?",
        (patient_id,),
    )
    meds = [dict(r) for r in cur.fetchall()]
    conn.close()

    lines = [
        f"ID: {p['id']}",
        f"Nome: {p['nome']}",
        f"Data de nascimento: {p['data_nascimento']}",
        f"Sexo: {p['sexo']}",
        f"Alergias: {p['alergias']}",
        f"Observações gerais: {p['observacoes']}",
        "",
        "Registros clínicos recentes:",
    ]
    for r in regs:
        lines.append(f"  - {r['data_registro']} [{r['tipo']}]: {r['resumo']}")
    lines.append("")
    lines.append("Medicações ativas (ilustrativas):")
    for m in meds:
        lines.append(f"  - {m['medicamento']} {m['dose'] or ''} (desde {m['inicio']})")
    return "\n".join(lines)


def fetch_all_patients_context_text(db_path: Path | None = None) -> str:
    """
    Monta um resumo textual com dados de **todos** os pacientes da base demo,
    para contextualizar o assistente sem filtrar por um único `patient_id`.
    """
    path = ensure_database(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id FROM pacientes ORDER BY id")
    ids = [int(r[0]) for r in cur.fetchall()]
    conn.close()

    if not ids:
        return "(Nenhum paciente na base demo.)"

    blocks: list[str] = []
    for pid in ids:
        blocks.append(f"--- Paciente {pid} ---")
        blocks.append(fetch_patient_context_text(pid, db_path=db_path))
        blocks.append("")
    return "\n".join(blocks).rstrip()


def run_sql_query(sql: str, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Executa SQL somente leitura (uso interno / agente)."""
    path = ensure_database(db_path)
    forbidden = ("insert", "update", "delete", "drop", "alter", "attach", "pragma")
    low = sql.strip().lower()
    if not low.startswith("select"):
        raise ValueError("Apenas consultas SELECT são permitidas neste demo.")
    if any(bad in low for bad in forbidden):
        raise ValueError("Consulta rejeitada por segurança.")
    engine = make_sqlalchemy_engine(path)
    with engine.connect() as c:
        result = c.execute(text(sql))
        cols = result.keys()
        return [dict(zip(cols, row)) for row in result.fetchall()]
