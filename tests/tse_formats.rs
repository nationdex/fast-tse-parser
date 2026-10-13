use fast_tse_parser::{
    BemCandidato, ConsultaCand, Dialect, Encoding, RecordIter, TseTable, VotacaoSecao,
};

const CONSULTA: &[u8] = b"DT_GERACAO;HH_GERACAO;ANO_ELEICAO;NR_TURNO;SG_UF;SG_UE;CD_CARGO;DS_CARGO;SQ_CANDIDATO;NR_CANDIDATO;NM_CANDIDATO;NM_URNA_CANDIDATO;SG_PARTIDO;DS_SITUACAO_CANDIDATO_TURNO;ST_REELEICAO\r\n\
12/08/2022;10:00:00;2022;1;BR;BR;1;PRESIDENTE;1001;13;\"SILVA; JO\xc3\x83O\";\"JO\xc3\x83O\";PT;APTO;N\r\n\
12/08/2022;10:00:00;2022;1;SP;SP;11;GOVERNADOR;1002;25;\"SOUZA\";\"SOUZA\";PSDB;APTO;S\r\n\
12/08/2022;10:00:00;2018;1;RJ;RJ;1;PRESIDENTE;1003;17;\"\"\"BOLSO\"\"\";\"BOLSO\";PL;APTO;N\r\n";

const VOTACAO: &[u8] =
    b"ANO_ELEICAO;SG_UF;NM_MUNICIPIO;NR_ZONA;NR_SECAO;SQ_CANDIDATO;NM_VOTAVEL;QT_VOTOS\r\n\
2022;SP;SAO PAULO;1;1;1001;\"SILVA; JO\xc3\x83O\";120\r\n\
2022;SP;SAO PAULO;1;1;1002;SOUZA;80\r\n\
2022;SP;SAO PAULO;1;2;1001;\"SILVA; JO\xc3\x83O\";5\r\n";

const BENS: &[u8] = b"ANO_ELEICAO;SQ_CANDIDATO;NM_CANDIDATO;DS_TIPO_BEM_CANDIDATO;DS_BEM_CANDIDATO;VR_BEM_CANDIDATO\r\n\
2022;1001;\"SILVA; JO\xc3\x83O\";\"Im\xc3\xb3vel\";\"Apartamento; pr\xc3\xa9dio\";1.234,56\r\n\
2022;1001;\"SILVA; JO\xc3\x83O\";Ve\xc3\xadculo;\"Carro\";30.000,00\r\n";

#[test]
fn consulta_cand_typed_access() {
    let doc = ConsultaCand::parse(CONSULTA).unwrap();
    let rows: Vec<_> = doc.iter().collect();
    assert_eq!(rows.len(), 3);

    assert_eq!(rows[0].sq_candidato(), Some(1001));
    assert_eq!(rows[0].nr_candidato(), Some(13));
    assert_eq!(
        rows[0].nm_candidato().unwrap().as_str(),
        Some("SILVA; JOÃO")
    );
    assert_eq!(rows[0].sg_uf().unwrap().as_str(), Some("BR"));
    assert_eq!(rows[0].ano_eleicao(), Some(2022));

    // Escaped quotes inside quoted name.
    assert_eq!(rows[2].nm_candidato().unwrap().as_str(), None);
    let mut out = Vec::new();
    rows[2].nm_candidato().unwrap().decode_into(&mut out);
    assert_eq!(out, b"\"BOLSO\"");
}

#[test]
fn election_cycle_discovery() {
    let doc = ConsultaCand::parse(CONSULTA).unwrap();
    let mut cycles = Vec::new();
    doc.election_cycles(|f| cycles.push(f.to_u32().unwrap()));
    assert_eq!(cycles, vec![2022, 2018]);
}

#[test]
fn votacao_totals() {
    let doc = VotacaoSecao::parse(VOTACAO).unwrap();
    assert_eq!(doc.total_votes(), 205);
    let first = doc.iter().next().unwrap();
    assert_eq!(first.nr_zona(), Some(1));
    assert_eq!(first.nr_secao(), Some(1));
    assert_eq!(first.nm_votavel().unwrap().as_str(), Some("SILVA; JOÃO"));
}

#[test]
fn bens_total_cents() {
    let doc = BemCandidato::parse(BENS).unwrap();
    assert_eq!(doc.total_cents(), 123_456 + 3_000_000);
    let mut name = String::new();
    doc.iter()
        .next()
        .unwrap()
        .row()
        .get_text("NM_CANDIDATO", &mut name);
    assert_eq!(name, "SILVA; JOÃO");
}

#[test]
fn latin1_rows_detected() {
    let data: &[u8] = b"NM;UF\n\"ANT\xc3\x82NIO JOSE\";SP\n";
    let table = TseTable::new(data).unwrap();
    assert_eq!(table.encoding(), Encoding::Utf8);

    let latin: &[u8] = b"NM;UF\n\"ANT\xd4NIO JOSE\";SP\n";
    let table = TseTable::new(latin).unwrap();
    assert_eq!(table.encoding(), Encoding::Latin1);
    let row = table.rows().next().unwrap();
    let mut name = String::new();
    row.get_text("NM", &mut name);
    assert_eq!(name, "ANTÔNIO JOSE");
}

#[test]
fn quoted_embedded_newline_and_sep() {
    let data = b"H;A;B\n1;\"x\ny;z\";w\n2;p;q\n";
    let recs: Vec<_> = RecordIter::new(data, Dialect::TSE)
        .map(|r| r.as_bytes().to_vec())
        .collect();
    assert_eq!(recs.len(), 3);
    assert_eq!(recs[1], b"1;\"x\ny;z\";w".to_vec());
    assert_eq!(recs[2], b"2;p;q".to_vec());
}

#[test]
fn header_lookup_by_name_is_layout_agnostic() {
    let doc = ConsultaCand::parse(CONSULTA).unwrap();
    assert!(doc.table().header().index_of("SQ_CANDIDATO").is_some());
    assert!(doc.table().header().index_of("NO_SUCH_COL").is_none());
}
