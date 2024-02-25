# Use the specified base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set the working directory to your project directory
WORKDIR ./

# Copy the contents of your project into the Docker image
COPY . .

# Create and activate Conda environment
# RUN conda create --name plm python=3.10
# SHELL ["conda", "run", "-n", "plm", "/bin/bash", "-c"]
# RUN conda activate plm

# Install Miniconda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
#     /bin/bash miniconda.sh -b -p /opt/conda && \
#     rm miniconda.sh
# ENV PATH="/opt/conda/bin:${PATH}"

# Install dependencies
# RUN cd protein_lm/modeling/models/libs/ && pip install -e causal-conv1d && pip install -e mamba && cd ../../../../
# RUN pip install transformers datasets accelerate evaluate pytest fair-esm biopython deepspeed
# RUN pip install -e .
# RUN pip install hydra-core --upgrade
# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# source "$HOME/.cargo/env"
# RUN pip install -e protein_lm/tokenizer/rust_trie


