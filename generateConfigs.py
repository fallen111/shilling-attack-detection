import os

def generateConfigFiles():
    attack_sizes = [0.03, 0.08, 0.13, 0.2]
    attacks = [ 'Hybrid', 'Random', 'RR']
    os.makedirs(f".\\config",exist_ok=True)
    for attack in attacks:
        for attack_size in attack_sizes:
            config_name = f"{attack}_Attack_attackSize_{int(attack_size*100)}"
            output_dir = f".\\output\\{config_name}"
            os.makedirs(output_dir,exist_ok=True)
            
            config_content = rf"""ratings=.\dataset\MovieLense\ratings_no_timestamp.txt
    ratings.setup=-columns 0 1 2
    social=.\\dataset\\filmtrust\\trust.txt
    social.setup=-columns 0 1 2
    attackSize={attack_size}
    fillerSize=0.05
    selectedSize=0.005
    targetCount=20
    targetScore=4.0
    threshold=3.0
    maxScore=4.0
    minScore=1.0
    minCount=5
    maxCount=50
    linkSize=0.001
    outputDir={output_dir}
    """
            with open(f".\\config\\{config_name}.txt", "w") as config_file:
                config_file.write(config_content)
