A deep learning-powered repository for generating personalized video content with user annotations. Utilizes state-of-the-art GANs to synthesize beautiful visuals

SyntheticMediaGenerator/
├── README.md              # Project description and instructions
├── improved_gan.py        # Main GAN model implementation
├── requirements.txt       # List of required packages and libraries
├── generated_images/      # Folder for storing generated images
│   ├── grid_epoch_*.png   # Grid images for specific epochs
│   └── single_epoch_*.png # Individual generated images for specific epochs
├── saved_models/          # Folder for storing saved model weights
│   ├── generator_epoch_*.pth
│   └── discriminator_epoch_*.pth
└── Images/                # Dataset folder containing images for training
    ├── class1/            # Subfolder for class 1 images
    └── class2/            # Subfolder for class 2 images

