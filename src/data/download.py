"""
Download viral genome sequences from NCBI
"""
import os
import time
from typing import List, Dict
from pathlib import Path
from Bio import Entrez, SeqIO
import pandas as pd
from tqdm import tqdm
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NCBIGenomeDownloader:
    """Download viral genomes from NCBI GenBank"""
    
    def __init__(self, email: str, api_key: str = None, output_dir: str = "data/raw"):
        """
        Initialize NCBI downloader
        
        Args:
            email: Your email for NCBI API
            api_key: NCBI API key (optional, increases rate limit)
            output_dir: Directory to save downloaded sequences
        """
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def search_viral_family(
        self, 
        family_name: str, 
        max_results: int = 200,
        complete_only: bool = True
    ) -> List[str]:
        """
        Search for viral genomes by family name
        
        Args:
            family_name: Taxonomic family name (e.g., "Coronaviridae")
            max_results: Maximum number of sequences to retrieve
            complete_only: Only download complete genomes
            
        Returns:
            List of GenBank IDs
        """
        # Construct search query
        query = f"{family_name}[Organism]"
        
        if complete_only:
            query += " AND complete genome[Title]"
        
        query += " AND 1000:50000[Sequence Length]"  # Filter by length
        
        logger.info(f"Searching NCBI for: {query}")
        
        try:
            # Search NCBI nucleotide database
            handle = Entrez.esearch(
                db="nucleotide",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            
            id_list = record["IdList"]
            logger.info(f"Found {len(id_list)} sequences for {family_name}")
            
            return id_list
            
        except Exception as e:
            logger.error(f"Error searching for {family_name}: {e}")
            return []
    
    def download_sequences(
        self, 
        id_list: List[str], 
        family_name: str,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Download sequences and metadata
        
        Args:
            id_list: List of GenBank IDs
            family_name: Virus family name for organizing files
            batch_size: Number of sequences per batch
            
        Returns:
            DataFrame with sequence metadata
        """
        metadata = []
        output_file = self.output_dir / f"{family_name.lower()}.fasta"
        
        # Download in batches to avoid API limits
        for i in tqdm(range(0, len(id_list), batch_size), desc=f"Downloading {family_name}"):
            batch_ids = id_list[i:i + batch_size]
            
            try:
                # Fetch sequences
                handle = Entrez.efetch(
                    db="nucleotide",
                    id=batch_ids,
                    rettype="fasta",
                    retmode="text"
                )
                
                # Parse and save
                records = list(SeqIO.parse(handle, "fasta"))
                handle.close()
                
                # Append to file
                with open(output_file, "a") as f:
                    SeqIO.write(records, f, "fasta")
                
                # Collect metadata
                for record in records:
                    metadata.append({
                        "id": record.id,
                        "description": record.description,
                        "family": family_name,
                        "length": len(record.seq),
                        "file": str(output_file)
                    })
                
                # Rate limiting (3 requests/sec without API key, 10 with)
                time.sleep(0.35 if Entrez.api_key else 1.0)
                
            except Exception as e:
                logger.error(f"Error downloading batch {i}: {e}")
                continue
        
        logger.info(f"Downloaded {len(metadata)} sequences for {family_name}")
        return pd.DataFrame(metadata)
    
    def download_all_families(
        self, 
        families: List[str], 
        samples_per_family: int = 200
    ) -> pd.DataFrame:
        """
        Download genomes for all specified virus families
        
        Args:
            families: List of virus family names
            samples_per_family: Max sequences per family
            
        Returns:
            Combined metadata DataFrame
        """
        all_metadata = []
        
        for family in families:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {family}")
            logger.info(f"{'='*50}")
            
            # Search for sequences
            id_list = self.search_viral_family(family, max_results=samples_per_family)
            
            if not id_list:
                logger.warning(f"No sequences found for {family}")
                continue
            
            # Download sequences
            metadata_df = self.download_sequences(id_list, family)
            all_metadata.append(metadata_df)
            
            # Save intermediate results
            metadata_df.to_csv(
                self.output_dir / f"{family.lower()}_metadata.csv",
                index=False
            )
        
        # Combine all metadata
        if all_metadata:
            combined_metadata = pd.concat(all_metadata, ignore_index=True)
            combined_metadata.to_csv(
                self.output_dir / "all_sequences_metadata.csv",
                index=False
            )
            logger.info(f"\nTotal sequences downloaded: {len(combined_metadata)}")
            return combined_metadata
        
        return pd.DataFrame()