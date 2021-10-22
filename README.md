# sistema-fgac-parzen

Desenvolvimento de sistema de segmentação de Acidente Vascular Cerebral Hemorrágico através de uma abordagem com Level Set utilizando estimação por janela de Parzen. 

Em Estatística, estimativa de densidade por Kernel (EDK) é uma forma não-paramétrica para estimar a Função densidade de probabilidade (FDP) de uma variável aleatória. Estimativa da densidade por Kernel é um problema fundamental de suavização de dados onde inferências sobre a população são feitas com base em uma amostra de dados finita. Em alguns campos, como o Processamento de sinais e Econometria é também denominado como o método da janela de Parzen-Rosenblatt, devido aos nomes de Emanuel Parzen e Murray Rosenblatt, que são creditados por criá-lo de forma independente em sua forma atual. (FONTE: [DUDA, R. O.; HART, P. E.; STORK, D. G. Pattern Classification] (https://libgen.is/book/index.php?md5=FDFDEA9D8171EF45F0B2EEA8030490D0))

> O **Acidente Vascular Cerebral (AVC)** acontece quando vasos que levam sangue ao cérebro entopem ou se rompem, provocando a paralisia da área cerebral que ficou sem circulação sanguínea. É uma doença que acomete mais os homens e é uma das principais causas de morte, incapacitação e internações em todo o mundo. (FONTE: [Ministério da Saúde](https://antigo.saude.gov.br/saude-de-a-z/acidente-vascular-cerebral-avc))

Quanto mais rápido for o diagnóstico e o tratamento do AVC, maiores serão as chances de recuperação completa. Desta forma, torna-se primordial ficar atento aos sinais e sintomas e procurar atendimento médico imediato.

Existem dois tipos de AVC, que ocorrem por motivos diferentes:

- **AVC hemorrágico (AVCh)**: O AVCh ocorre quando há rompimento de um vaso cerebral, provocando hemorragia. Esta hemorragia pode acontecer dentro do tecido cerebral ou na superfície entre o cérebro e a meninge. É responsável por 15% de todos os casos de AVC, mas pode causar a morte com mais frequência do que o AVC isquêmico.

- **AVC isquêmico (AVCi)**: O AVC isquêmico ocorre quando há obstrução de uma artéria, impedindo a passagem de oxigênio para células cerebrais, que acabam morrendo. Essa obstrução pode acontecer devido a um trombo (trombose) ou a um êmbolo (embolia). O AVC isquêmico é o mais comum e representa 85% de todos os casos.



