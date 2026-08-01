# Third-party data and implementation references

This repository does not redistribute the HUST raw archive. Obtain HUST version 2 from the registered Mendeley Data record, DOI `10.17632/nsc7hnsg4s.2`. The record identifies the dataset contributors as Ye Yuan, Guijun Ma, and Songpei Xu and displays a CC BY 4.0 licence; users should verify and comply with the repository terms at the time of use.

The HUST per-cycle discharge-capacity parser follows the public Microsoft BatteryML HUST preprocessing approach, including negative-current time integration and the source-cell `7-5` initial-cycle rule. The referenced BatteryML source is licensed under the MIT License. Source reference:

https://github.com/microsoft/BatteryML/blob/main/batteryml/preprocess/preprocess_HUST.py

The bundled XJTU Batch 1+3 prepared curves are study inputs previously supplied for the MSTT_RUL analysis. Users remain responsible for complying with the original XJTU dataset terms when redistributing or publishing derived artifacts.

Python dependencies remain subject to their respective licenses. See `requirements.txt` for the runtime dependency names.
