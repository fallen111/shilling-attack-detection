from averageAttack import AverageAttack
# from bandwagonAttack import BandWagonAttack
# from randomAttack import RandomAttack
# from hybridAttack import HybridAttack
# from RR_Attack import RR_Attack 

attack = AverageAttack('config.conf')
attack.insertSpam()
# attack.farmLink()
attack.generateLabels('labels.txt')
attack.generateProfiles('profiles.txt')
# attack.generateSocialConnections('relations.txt')