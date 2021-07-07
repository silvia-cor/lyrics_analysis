from classification import round_of_kcv, binary_classification


round_of_kcv(domain='genres', feat='phonetics')
binary_classification(domain='artists', feat='all', save=False)
