# HandVerify
HandVerify è un sistema di verifica biometrica basato sulla scrittura a mano, progettato per determinare se due testi manoscritti (anche con contenuti diversi) siano stati prodotti dalla stessa persona.


##  Comando

QT_PLUGIN_PATH="" LD_LIBRARY_PATH="" uv run python webcam_demo.py --camera 4 --checkpoint ../../results/demo/resnet18_contrastive_mixed_iam_rimes_stratified_best.pth --threshold eer