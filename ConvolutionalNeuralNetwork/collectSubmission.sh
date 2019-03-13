rm -f assignment5.zip
zip -r assignment5.zip . -x "*lib/datasets*" "*.ipynb_checkpoints*" "*.idea*" "*collectSubmission.sh" "*requirements.txt" "*.pyc"
