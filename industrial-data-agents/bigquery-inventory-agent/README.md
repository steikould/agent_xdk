# BigQuery Inventory Agent

This agent takes an inventory of all BigQuery datasets and tables in a GCP project. It interacts with a data engineer through an LLM to compare the data in the BigQuery datalake with documents in a SharePoint folder. The goal is to understand the data landscape and identify discrepancies between on-prem/SaaS application data and the data in the datalake.
