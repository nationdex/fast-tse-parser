//! Typed views over TSE tables: `consulta_cand`, `votacao_secao`, `bem_candidato`.
//!
//! Column resolution is by header name, so minor layout drift between election
//! years does not break the parser. All accessors borrow from the mmap; numbers
//! parse through stack buffers only.

use crate::header::{fnv1a, Header};

const H_ANO_ELEICAO: u64 = fnv1a(b"ANO_ELEICAO");
const H_NR_TURNO: u64 = fnv1a(b"NR_TURNO");
const H_SG_UF: u64 = fnv1a(b"SG_UF");
const H_SG_UE: u64 = fnv1a(b"SG_UE");
const H_DS_CARGO: u64 = fnv1a(b"DS_CARGO");
const H_SQ_CANDIDATO: u64 = fnv1a(b"SQ_CANDIDATO");
const H_NR_CANDIDATO: u64 = fnv1a(b"NR_CANDIDATO");
const H_NM_CANDIDATO: u64 = fnv1a(b"NM_CANDIDATO");
const H_NM_URNA_CANDIDATO: u64 = fnv1a(b"NM_URNA_CANDIDATO");
const H_SG_PARTIDO: u64 = fnv1a(b"SG_PARTIDO");
const H_DS_SIT_TURNO: u64 = fnv1a(b"DS_SITUACAO_CANDIDATO_TURNO");
const H_DS_SIT_CAND: u64 = fnv1a(b"DS_SITUACAO_CANDIDATURA");
const H_ST_REELEICAO: u64 = fnv1a(b"ST_REELEICAO");
const H_NM_MUNICIPIO: u64 = fnv1a(b"NM_MUNICIPIO");
const H_NR_ZONA: u64 = fnv1a(b"NR_ZONA");
const H_NR_SECAO: u64 = fnv1a(b"NR_SECAO");
const H_NM_VOTAVEL: u64 = fnv1a(b"NM_VOTAVEL");
const H_QT_VOTOS: u64 = fnv1a(b"QT_VOTOS");
const H_DS_TIPO_BEM: u64 = fnv1a(b"DS_TIPO_BEM_CANDIDATO");
const H_DS_BEM: u64 = fnv1a(b"DS_BEM_CANDIDATO");
const H_VR_BEM: u64 = fnv1a(b"VR_BEM_CANDIDATO");

use crate::latin1::{self, Encoding};
use crate::scan::{Dialect, Field, FieldIter, RecordIter, MAX_COLS};
use crate::simd::Backend;
use std::cell::Cell;

/// Generic parsed table: header map + row iterator.
pub struct TseTable<'a> {
    data: &'a [u8],
    dialect: Dialect,
    encoding: Encoding,
    backend: Backend,
    header: Header,
    body_start: usize,
}

impl<'a> TseTable<'a> {
    pub fn new(data: &'a [u8]) -> Result<TseTable<'a>, crate::Error> {
        Self::with_dialect(data, Dialect::TSE)
    }

    pub fn with_dialect(data: &'a [u8], dialect: Dialect) -> Result<TseTable<'a>, crate::Error> {
        let encoding = latin1::detect(data);
        let backend = Backend::detect();
        let mut records = RecordIter::with_backend(data, backend, dialect);
        let head = records.next().ok_or(crate::Error::EmptyInput)?;
        let header = Header::parse(head, dialect);
        Ok(TseTable {
            data,
            dialect,
            encoding,
            backend,
            header,
            body_start: head.end(),
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    pub fn dialect(&self) -> Dialect {
        self.dialect
    }

    /// Iterate data rows (header excluded).
    pub fn rows<'h>(&'h self) -> Rows<'a, 'h> {
        Rows {
            inner: RecordIter::with_backend(
                &self.data[self.body_start..],
                self.backend,
                self.dialect,
            ),
            table: self,
        }
    }

    /// Zero-alloc distinct-values callback over one column (e.g. election
    /// cycle discovery). Values are fed as raw fields.
    pub fn distinct_fields(&self, col: &str, f: impl FnMut(Field<'a>)) {
        self.distinct_fields_h(crate::header::fnv1a(col.as_bytes()), f)
    }

    /// `distinct_fields` with a precomputed column-name hash.
    pub fn distinct_fields_h(&self, hash: u64, mut f: impl FnMut(Field<'a>)) {
        let Some(idx) = self.header.index_of_hash(hash) else {
            return;
        };
        let mut seen: [Option<u64>; 32] = [None; 32];
        for row in self.rows() {
            let Some(field) = row.get_idx(idx) else {
                continue;
            };
            let hash = crate::header::fnv1a(field.as_bytes());
            let slot = (hash as usize) % seen.len();
            let mut probe = slot;
            loop {
                match seen[probe] {
                    None => {
                        seen[probe] = Some(hash);
                        f(field);
                        break;
                    }
                    Some(h) if h == hash => break,
                    Some(_) => probe = (probe + 1) % seen.len(),
                }
            }
        }
    }
}

/// Iterator over data rows.
pub struct Rows<'a, 'h> {
    inner: RecordIter<'a>,
    table: &'h TseTable<'a>,
}

impl<'a, 'h> Iterator for Rows<'a, 'h> {
    type Item = Row<'a, 'h>;

    fn next(&mut self) -> Option<Row<'a, 'h>> {
        let rec = self.inner.next()?;
        Some(Row::new(rec.as_bytes(), self.table))
    }
}

/// One data row. Field access is lazy: the record is scanned forward from a
/// cached cursor, so sequential column reads cost one pass, no allocation.
pub struct Row<'a, 'h> {
    rec: &'a [u8],
    table: &'h TseTable<'a>,
    /// (fields scanned, byte offset where field `scanned` starts).
    cursor: Cell<(u16, usize)>,
}

impl Clone for Row<'_, '_> {
    fn clone(&self) -> Self {
        Row {
            rec: self.rec,
            table: self.table,
            cursor: Cell::new(self.cursor.get()),
        }
    }
}

impl<'a, 'h> Row<'a, 'h> {
    pub fn new(rec_bytes: &'a [u8], table: &'h TseTable<'a>) -> Row<'a, 'h> {
        Row {
            rec: rec_bytes,
            table,
            cursor: Cell::new((0, 0)),
        }
    }

    pub fn as_bytes(&self) -> &'a [u8] {
        self.rec
    }

    /// Number of fields in the record. Scans the whole record once.
    pub fn len(&self) -> u16 {
        let mut n = 0u16;
        for _ in self.fields() {
            n += 1;
        }
        n
    }

    pub fn is_empty(&self) -> bool {
        self.rec.is_empty()
    }

    /// Field iterator over the whole record (fresh quote state).
    pub fn fields(&self) -> FieldIter<'a> {
        FieldIter::with_backend(self.rec, self.table.backend, self.table.dialect)
    }

    /// Split into at most `MAX_COLS` field slices. Returns field count.
    pub fn split(&self, buf: &mut [&'a [u8]; MAX_COLS]) -> usize {
        let mut n = 0;
        for field in self.fields() {
            if n == MAX_COLS {
                break;
            }
            buf[n] = field.as_bytes();
            n += 1;
        }
        n
    }

    pub fn raw(&self, idx: u16) -> Option<&'a [u8]> {
        self.get_idx(idx).map(|f| f.as_bytes())
    }

    pub fn get_idx(&self, idx: u16) -> Option<Field<'a>> {
        let (scanned, pos) = self.cursor.get();
        let (start_byte, skip) = if idx < scanned {
            // Out-of-order read: restart from the beginning.
            (0usize, idx)
        } else {
            (pos.min(self.rec.len()), idx - scanned)
        };
        let mut it = FieldIter::with_backend(
            &self.rec[start_byte..],
            self.table.backend,
            self.table.dialect,
        );
        let mut skip = skip;
        while skip > 0 {
            it.next()?;
            skip -= 1;
        }
        let field = it.next()?;
        self.cursor.set((idx + 1, start_byte + it.pos()));
        Some(field)
    }

    pub fn get(&self, name: &str) -> Option<Field<'a>> {
        self.get_idx(self.table.header.index_of(name)?)
    }

    /// Lookup by precomputed FNV-1a hash of the column name (const-friendly).
    pub fn get_h(&self, hash: u64) -> Option<Field<'a>> {
        self.get_idx(self.table.header.index_of_hash(hash)?)
    }

    pub fn get_u32(&self, name: &str) -> Option<u32> {
        self.get(name)?.to_u32()
    }

    pub fn get_u64(&self, name: &str) -> Option<u64> {
        self.get(name)?.to_u64()
    }

    pub fn get_u32_h(&self, hash: u64) -> Option<u32> {
        self.get_h(hash)?.to_u32()
    }

    pub fn get_u64_h(&self, hash: u64) -> Option<u64> {
        self.get_h(hash)?.to_u64()
    }

    /// Decode a named field to text honouring table encoding. Materialising
    /// path: may allocate (all core parsing stays zero-copy).
    pub fn get_text(&self, name: &str, out: &mut String) -> bool {
        match self.get(name) {
            Some(field) => {
                let mut buf = Vec::new();
                field.decode_into(&mut buf);
                latin1::decode_text(&buf, self.table.encoding, out);
                true
            }
            None => false,
        }
    }

    pub fn encoding(&self) -> Encoding {
        self.table.encoding
    }

    pub fn dialect(&self) -> Dialect {
        self.table.dialect
    }
}

// ---------------------------------------------------------------------------
// consulta_cand
// ---------------------------------------------------------------------------

/// Candidate registration record (`consulta_cand`).
#[derive(Clone)]
pub struct Candidato<'a, 'h>(Row<'a, 'h>);

impl<'a, 'h> Candidato<'a, 'h> {
    pub fn row(&self) -> &Row<'a, 'h> {
        &self.0
    }

    pub fn ano_eleicao(&self) -> Option<u32> {
        self.0.get_u32_h(H_ANO_ELEICAO)
    }

    pub fn nr_turno(&self) -> Option<u32> {
        self.0.get_u32_h(H_NR_TURNO)
    }

    pub fn sg_uf(&self) -> Option<Field<'a>> {
        self.0.get_h(H_SG_UF)
    }

    pub fn sg_ue(&self) -> Option<Field<'a>> {
        self.0.get_h(H_SG_UE)
    }

    pub fn ds_cargo(&self) -> Option<Field<'a>> {
        self.0.get_h(H_DS_CARGO)
    }

    pub fn sq_candidato(&self) -> Option<u64> {
        self.0.get_u64_h(H_SQ_CANDIDATO)
    }

    pub fn nr_candidato(&self) -> Option<u32> {
        self.0.get_u32_h(H_NR_CANDIDATO)
    }

    pub fn nm_candidato(&self) -> Option<Field<'a>> {
        self.0.get_h(H_NM_CANDIDATO)
    }

    pub fn nm_urna_candidato(&self) -> Option<Field<'a>> {
        self.0.get_h(H_NM_URNA_CANDIDATO)
    }

    pub fn sg_partido(&self) -> Option<Field<'a>> {
        self.0.get_h(H_SG_PARTIDO)
    }

    pub fn ds_sit_cand_tot(&self) -> Option<Field<'a>> {
        self.0
            .get_h(H_DS_SIT_TURNO)
            .or_else(|| self.0.get_h(H_DS_SIT_CAND))
    }

    pub fn st_reeleicao(&self) -> Option<Field<'a>> {
        self.0.get_h(H_ST_REELEICAO)
    }
}

pub struct ConsultaCand<'a>(TseTable<'a>);

impl<'a> ConsultaCand<'a> {
    pub fn parse(data: &'a [u8]) -> Result<ConsultaCand<'a>, crate::Error> {
        Ok(ConsultaCand(TseTable::new(data)?))
    }

    pub fn table(&self) -> &TseTable<'a> {
        &self.0
    }

    pub fn iter<'h>(&'h self) -> impl Iterator<Item = Candidato<'a, 'h>> + 'h {
        self.0.rows().map(Candidato)
    }

    /// Election cycle discovery: distinct `ANO_ELEICAO` values, zero alloc.
    pub fn election_cycles(&self, f: impl FnMut(Field<'a>)) {
        self.0.distinct_fields_h(H_ANO_ELEICAO, f)
    }
}

// ---------------------------------------------------------------------------
// votacao_secao
// ---------------------------------------------------------------------------

/// Per-section vote tally (`votacao_secao`).
#[derive(Clone)]
pub struct VotoSecao<'a, 'h>(Row<'a, 'h>);

impl<'a, 'h> VotoSecao<'a, 'h> {
    pub fn row(&self) -> &Row<'a, 'h> {
        &self.0
    }

    pub fn ano_eleicao(&self) -> Option<u32> {
        self.0.get_u32_h(H_ANO_ELEICAO)
    }

    pub fn sg_uf(&self) -> Option<Field<'a>> {
        self.0.get_h(H_SG_UF)
    }

    pub fn nm_municipio(&self) -> Option<Field<'a>> {
        self.0.get_h(H_NM_MUNICIPIO)
    }

    pub fn nr_zona(&self) -> Option<u32> {
        self.0.get_u32_h(H_NR_ZONA)
    }

    pub fn nr_secao(&self) -> Option<u32> {
        self.0.get_u32_h(H_NR_SECAO)
    }

    pub fn sq_candidato(&self) -> Option<u64> {
        self.0.get_u64_h(H_SQ_CANDIDATO)
    }

    pub fn nm_votavel(&self) -> Option<Field<'a>> {
        self.0.get_h(H_NM_VOTAVEL)
    }

    pub fn qt_votos(&self) -> Option<u32> {
        self.0.get_u32_h(H_QT_VOTOS)
    }
}

pub struct VotacaoSecao<'a>(TseTable<'a>);

impl<'a> VotacaoSecao<'a> {
    pub fn parse(data: &'a [u8]) -> Result<VotacaoSecao<'a>, crate::Error> {
        Ok(VotacaoSecao(TseTable::new(data)?))
    }

    pub fn table(&self) -> &TseTable<'a> {
        &self.0
    }

    pub fn iter<'h>(&'h self) -> impl Iterator<Item = VotoSecao<'a, 'h>> + 'h {
        self.0.rows().map(VotoSecao)
    }

    /// Sum `QT_VOTOS` with zero intermediate state.
    pub fn total_votes(&self) -> u64 {
        self.iter()
            .filter_map(|v| v.qt_votos())
            .map(u64::from)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// bem_candidato
// ---------------------------------------------------------------------------

/// Declared asset of a candidate (`bem_candidato`).
#[derive(Clone)]
pub struct Bem<'a, 'h>(Row<'a, 'h>);

impl<'a, 'h> Bem<'a, 'h> {
    pub fn row(&self) -> &Row<'a, 'h> {
        &self.0
    }

    pub fn sq_candidato(&self) -> Option<u64> {
        self.0.get_u64_h(H_SQ_CANDIDATO)
    }

    pub fn nm_candidato(&self) -> Option<Field<'a>> {
        self.0.get_h(H_NM_CANDIDATO)
    }

    pub fn ds_tipo_bem(&self) -> Option<Field<'a>> {
        self.0.get_h(H_DS_TIPO_BEM)
    }

    pub fn ds_bem(&self) -> Option<Field<'a>> {
        self.0.get_h(H_DS_BEM)
    }

    pub fn vr_bem_cents(&self) -> Option<u64> {
        self.0.get_h(H_VR_BEM)?.to_cents()
    }
}

pub struct BemCandidato<'a>(TseTable<'a>);

impl<'a> BemCandidato<'a> {
    pub fn parse(data: &'a [u8]) -> Result<BemCandidato<'a>, crate::Error> {
        Ok(BemCandidato(TseTable::new(data)?))
    }

    pub fn table(&self) -> &TseTable<'a> {
        &self.0
    }

    pub fn iter<'h>(&'h self) -> impl Iterator<Item = Bem<'a, 'h>> + 'h {
        self.0.rows().map(Bem)
    }

    /// Total declared assets in cents, zero alloc.
    pub fn total_cents(&self) -> u64 {
        self.iter().filter_map(|b| b.vr_bem_cents()).sum()
    }
}
