# Türkiye elektrik fiyatları — literatür tarama raporu

`python scripts/lit_search.py report` · 2026-08-22

**Havuz:** 3973 iş · **kapsam içi:** 102 (Türkiye ∧ elektrik fiyatı) · 12 tohum · 12 kayıtlı sorgu

**Sıralama = sağlamlık vekili**, konu değil: `0,35·atıf hızı + 0,25·dergi h-endeksi + 0,25·tohum bağı + 0,15·güncellik` (hepsi kapsam içi havuzda yüzdelik).

> Bunlar tam metin okumadan görülebilen VEKİLLER. Test seti uzunluğu, bölme biçimi ve girdilerin tahmin anında mevcut olup olmadığı ancak tam metinde görülüyor — `extract` formu onun için var.

## En sağlam işler (konu farketmeksizin)

| # | Güç | Yıl | Atıf/yıl | Dergi h | Bağ | OA | Özet | Konu | Başlık |
|--:|--:|--:|--:|--:|--:|:-:|:-:|---|---|
| 1 | 0.935 | 2021 | 7.8 | 356 | 4 | · | ✖ | yenilenebilir-etkisi | [The impact of variable renewable energy technologies on elec](https://doi.org/10.1016/j.enpol.2020.112093) |
| 2 | 0.858 | 2020 | 8.7 | 356 | 3 | · | ✖ | yenilenebilir-etkisi | [Variable renewable energy technologies in the Turkish electr](https://doi.org/10.1016/j.enpol.2020.111660) |
| 3 | 0.855 | 2022 | 7.2 | 356 | 1 | · | ✖ | politika/tavan | [Price spikes, temporary price caps, and welfare effects of r](https://doi.org/10.1016/j.enpol.2022.112816) |
| 4 | 0.833 | 2017 | 8.9 | 601 | 1 | ✔ | ✔ | tahmin | [Artificial neural network and SARIMA based models for power ](https://doi.org/10.1371/journal.pone.0175915) |
| 5 | 0.828 | 2023 | 4.5 | 348 | 1 | · | ✖ | tahmin | [Electricity price estimation using deep learning approaches:](https://doi.org/10.1016/j.eswa.2023.120026) |
| 6 | 0.825 | 2019 | 4.1 | 356 | 4 | ✔ | ✖ | yenilenebilir-etkisi | [The merit order effect of wind and river type hydroelectrici](https://doi.org/10.1016/j.enpol.2019.07.006) |
| 7 | 0.788 | 2024 | 4.0 | 87 | 2 | · | ✖ | yenilenebilir-etkisi | [Merit-order of dispatchable and variable renewable energy so](https://doi.org/10.1016/j.jup.2024.101758) |
| 8 | 0.786 | 2021 | 17.7 | 386 | 0 | ✔ | ✔ | tahmin | [Data augmentation for time series regression: Applying trans](https://doi.org/10.1016/j.apenergy.2021.117695) |
| 9 | 0.784 | 2022 | 3.6 | 228 | 1 | ✔ | ✔ | yenilenebilir-etkisi | [Modelling the Potential Impacts of Nuclear Energy and Renewa](https://doi.org/10.3390/en15041392) |
| 10 | 0.782 | 2023 | 7.8 | 451 | 0 | ✔ | ✔ | tahmin | [Wind power plants hybridised with solar power: A generation ](https://doi.org/10.1016/j.jclepro.2023.138793) |
| 11 | 0.777 | 2021 | 10.5 | 372 | 0 | ✔ | ✔ | tahmin | [Multi-Horizon Electricity Load and Price Forecasting Using a](https://doi.org/10.1109/access.2021.3086039) |
| 12 | 0.747 | 2022 | 9.8 | 318 | 0 | · | ✖ | diğer | [The role of data frequency and method selection in electrici](https://doi.org/10.1016/j.renene.2021.12.136) |
| 13 | 0.741 | 2022 | 23.0 | 228 | 0 | ✔ | ✔ | yenilenebilir-etkisi | [Electric Vehicles and Vehicle–Grid Interaction in the Turkis](https://doi.org/10.3390/en15218218) |
| 14 | 0.731 | 2024 | 1.0 | 318 | 2 | · | ✖ | yenilenebilir-etkisi | [Measuring the long-term impact of wind, run-of-river, solar ](https://doi.org/10.1016/j.renene.2024.122292) |
| 15 | 0.723 | 2009 | 5.9 | 182 | 2 | · | ✔ | maliyet-geçişi | [Long‐run relations in European electricity prices](https://doi.org/10.1002/jae.1095) |
| 16 | 0.718 | 2018 | 2.4 | 356 | 1 | ✔ | ✖ | tahmin | [Managing electricity price modeling risk via ensemble foreca](https://doi.org/10.1016/j.enpol.2018.08.053) |
| 17 | 0.713 | 2016 | 6.5 | 136 | 2 | ✔ | ✔ | yenilenebilir-etkisi | [The Impact of RES in the Italian Day-Ahead and Balancing Mar](https://doi.org/10.5547/01956574.37.si2.agia) |
| 18 | 0.684 | 2023 | 3.5 | 292 | 0 | ✔ | ✔ | politika/tavan | [Navigating the crisis: Fuel price caps in the Australian nat](https://doi.org/10.1016/j.eneco.2023.107237) |
| 19 | 0.668 | 2022 | 4.4 | 154 | 0 | ✔ | ✔ | tahmin | [A Data-Driven Model to Forecast Multi-Step Ahead Time Series](https://doi.org/10.3390/electronics11101524) |
| 20 | 0.667 | 2023 | 1.0 | 287 | 1 | ✔ | ✔ | politika/tavan | [A dynamic multi-level iterative algorithm for clearing Europ](https://doi.org/10.1080/01605682.2023.2210182) |
| 21 | 0.666 | 2020 | 1.7 | 99 | 2 | · | ✖ | tahmin | [Electricity Day-Ahead Market Price Forecasting by Using Arti](https://doi.org/10.1007/s13369-020-04349-1) |
| 22 | 0.662 | 2021 | 3.2 | 228 | 0 | ✔ | ✔ | diğer | [Determination of Price Zones during Transition from Uniform ](https://doi.org/10.3390/en14041014) |
| 23 | 0.660 | 2022 | 5.4 | 113 | 0 | ✔ | ✔ | tahmin | [Short‐term electricity price forecasting based on graph conv](https://doi.org/10.1049/rpg2.12413) |
| 24 | 0.651 | 2019 | 2.1 | 153 | 1 | · | ✔ | diğer | [Optimizing Day-Ahead Electricity Market Prices: Increasing t](https://doi.org/10.1287/msom.2018.0767) |
| 25 | 0.649 | 2022 | 1.6 | 62 | 1 | ✔ | ✔ | yenilenebilir-etkisi | [Optimal Bidding Strategy of a Pumped Hydro Energy Storage In](https://doi.org/10.1155/2022/6261558) |
| 26 | 0.646 | 2024 | 1.7 | 48 | 1 | ✔ | ✔ | yenilenebilir-etkisi | [Impact of Renewable Energy Resources on the Turkish Power Ma](https://doi.org/10.32479/ijeep.16204) |
| 27 | 0.634 | 2024 | 0.0 | 448 | 2 | ✔ | ✖ | tahmin | [Explainable Forecast of Electricity Market Price Using Machi](https://doi.org/10.2139/ssrn.4894108) |
| 28 | 0.634 | 2025 | 0.0 | 448 | 2 | ✔ | ✖ | tahmin | [Impact of Market Factors on Day-Ahead Electricity Prices: Ex](https://doi.org/10.2139/ssrn.5472209) |
| 29 | 0.634 | 2019 | 6.2 | 258 | 0 | ✔ | ✔ | yenilenebilir-etkisi | [A Distribution Market Clearing Mechanism for Renewable Gener](https://doi.org/10.1109/tii.2019.2896346) |
| 30 | 0.631 | 2019 | 2.0 | 109 | 1 | ✔ | ✔ | diğer | [An analysis of price spikes and deviations in the deregulate](https://doi.org/10.1016/j.esr.2019.100376) |

📖 = derleme/survey — alana giriş için önce bunlar.

## Belirsiz — elle kontrol gerekli (17 iş)

Bu işlerin özeti **OpenAlex, Crossref ve Semantic Scholar'ın üçünde de yok** (Elsevier özet paylaşmıyor), başlıklarında da ülke adı geçmiyor. Türkiye çalışması olup olmadıkları metinden anlaşılamıyor; listeye sadece **alıntı grafiği yakınlığı** (≥2 tohumla bağ) ile girdiler. Grafik yakınlığı 'Türkiye' demek DEĞİL — Türkiye çalışmaları genel literatürü zaten alıntılıyor. Bu yüzden ana sıralamaya katılmadılar.

| Bağ | Yıl | Dergi | Başlık |
|--:|--:|---|---|
| 3 | 2019 | Energy Economics | [Assessing the impact of renewable energy sources on the electricity pr](https://doi.org/10.1016/j.eneco.2019.104532) |
| 3 | 2018 | Energy Policy | [Greener, more integrated, and less volatile? A quantile regression ana](https://doi.org/10.1016/j.enpol.2018.10.017) |
| 3 | 2017 | Energy Policy | [An analysis of the decline of electricity spot prices in Europe: Who i](https://doi.org/10.1016/j.enpol.2017.04.034) |
| 3 | 2017 | Energy Economics | [Revitalising the wind power induced merit order effect to reduce whole](https://doi.org/10.1016/j.eneco.2017.08.003) |
| 3 | 2017 | Energy | [Nonlinear empirical pricing in electricity markets using fundamental w](https://doi.org/10.1016/j.energy.2017.07.181) |
| 3 | 2016 | Energy | [Modeling the UK electricity price distributions using quantile regress](https://doi.org/10.1016/j.energy.2016.02.025) |
| 3 | 2006 | Electric Power Systems Res | [Short-term electricity prices forecasting in a competitive market: A n](https://doi.org/10.1016/j.epsr.2006.09.022) |
| 2 | 2025 | Computational Economics | [Electricity Price Prediction using Artificial Neural Network Models: A](https://doi.org/10.1007/s10614-025-11063-3) |
| 2 | 2025 | SSRN Electronic Journal | [Economic Impacts of Renewable Energy Adoption in Emerging Economies: E](https://doi.org/10.2139/ssrn.5382332) |
| 2 | 2023 | Energy Policy | [Market failure or politics? Understanding the motives behind regulator](https://doi.org/10.1016/j.enpol.2023.113647) |
| 2 | 2022 | Renewable Energy | [Analyzing the asymmetric impacts of renewables on wholesale electricit](https://doi.org/10.1016/j.renene.2022.05.116) |
| 2 | 2019 | Energy Policy | [The impact of renewable energy forecast errors on imbalance volumes an](https://doi.org/10.1016/j.enpol.2019.06.035) |
| 2 | 2017 | Applied Energy | [A bat optimized neural network and wavelet transform approach for shor](https://doi.org/10.1016/j.apenergy.2017.10.058) |
| 2 | 2014 | Energy Policy | [The merit-order effect in the Italian power market: The impact of sola](https://doi.org/10.1016/j.enpol.2014.11.038) |
| 2 | 2013 | Energy Economics | [Renewable generation and electricity prices: Taking stock and new evid](https://doi.org/10.1016/j.eneco.2013.09.011) |
| 2 | 2008 | Energy Economics | [Short term forecasting of electricity prices for MISO hubs: Evidence f](https://doi.org/10.1016/j.eneco.2008.06.003) |
| 2 | 2008 | International Journal of F | [Forecasting spot electricity prices: A comparison of parametric and se](https://doi.org/10.1016/j.ijforecast.2008.08.004) |

## Kesin kapsam — başlığında Türkiye geçenler (77 iş)

Ana sıralamada, özeti bir ülke listesinde 'Turkey' geçtiği için giren birkaç yabancı çalışma olabiliyor. Bu liste o gürültüden arınmış: başlığın kendisi Türkiye diyor. **Tez kaynakçasının çekirdeği burası.**

| Güç | Yıl | Konu | Dergi | Başlık |
|--:|--:|---|---|---|
| 0.935 | 2021 | yenilenebilir-etkisi | Energy Policy | [The impact of variable renewable energy technologies on electricity mark](https://doi.org/10.1016/j.enpol.2020.112093) |
| 0.858 | 2020 | yenilenebilir-etkisi | Energy Policy | [Variable renewable energy technologies in the Turkish electricity market](https://doi.org/10.1016/j.enpol.2020.111660) |
| 0.833 | 2017 | tahmin | PLoS ONE | [Artificial neural network and SARIMA based models for power load forecas](https://doi.org/10.1371/journal.pone.0175915) |
| 0.828 | 2023 | tahmin | Expert Systems with Applic | [Electricity price estimation using deep learning approaches: An empirica](https://doi.org/10.1016/j.eswa.2023.120026) |
| 0.825 | 2019 | yenilenebilir-etkisi | Energy Policy | [The merit order effect of wind and river type hydroelectricity generatio](https://doi.org/10.1016/j.enpol.2019.07.006) |
| 0.788 | 2024 | yenilenebilir-etkisi | Utilities Policy | [Merit-order of dispatchable and variable renewable energy sources in Tur](https://doi.org/10.1016/j.jup.2024.101758) |
| 0.784 | 2022 | yenilenebilir-etkisi | Energies | [Modelling the Potential Impacts of Nuclear Energy and Renewables in the ](https://doi.org/10.3390/en15041392) |
| 0.747 | 2022 | diğer | Renewable Energy | [The role of data frequency and method selection in electricity price est](https://doi.org/10.1016/j.renene.2021.12.136) |
| 0.741 | 2022 | yenilenebilir-etkisi | Energies | [Electric Vehicles and Vehicle–Grid Interaction in the Turkish Electricit](https://doi.org/10.3390/en15218218) |
| 0.718 | 2018 | tahmin | Energy Policy | [Managing electricity price modeling risk via ensemble forecasting: The c](https://doi.org/10.1016/j.enpol.2018.08.053) |
| 0.668 | 2022 | tahmin | Electronics | [A Data-Driven Model to Forecast Multi-Step Ahead Time Series of Turkish ](https://doi.org/10.3390/electronics11101524) |
| 0.667 | 2023 | politika/tavan | Journal of the Operational | [A dynamic multi-level iterative algorithm for clearing European electric](https://doi.org/10.1080/01605682.2023.2210182) |
| 0.666 | 2020 | tahmin | Arabian Journal for Scienc | [Electricity Day-Ahead Market Price Forecasting by Using Artificial Neura](https://doi.org/10.1007/s13369-020-04349-1) |
| 0.662 | 2021 | diğer | Energies | [Determination of Price Zones during Transition from Uniform to Zonal Ele](https://doi.org/10.3390/en14041014) |
| 0.646 | 2024 | yenilenebilir-etkisi | International Journal of E | [Impact of Renewable Energy Resources on the Turkish Power Market](https://doi.org/10.32479/ijeep.16204) |
| 0.634 | 2024 | tahmin | SSRN Electronic Journal | [Explainable Forecast of Electricity Market Price Using Machine Learning ](https://doi.org/10.2139/ssrn.4894108) |
| 0.634 | 2025 | tahmin | SSRN Electronic Journal | [Impact of Market Factors on Day-Ahead Electricity Prices: Explainable Ma](https://doi.org/10.2139/ssrn.5472209) |
| 0.631 | 2019 | diğer | Energy Strategy Reviews | [An analysis of price spikes and deviations in the deregulated Turkish po](https://doi.org/10.1016/j.esr.2019.100376) |
| 0.630 | 2011 | diğer | Renewable and Sustainable  | [Financial optimization in the Turkish electricity market: Markowitz's me](https://doi.org/10.1016/j.rser.2011.06.018) |
| 0.625 | 2015 | tahmin | TURKISH JOURNAL OF ELECTRI | [Forecasting the day-ahead price in electricity balancing and settlement ](https://doi.org/10.3906/elk-1212-136) |
| 0.619 | 2023 | yenilenebilir-etkisi | Fiscaoeconomia | [The Effects of Electricity Generation from Solar and Wind Energy on the ](https://doi.org/10.25295/fsecon.1215578) |
| 0.614 | 2018 | tahmin | Emerging Markets Finance a | [Performance of Electricity Price Forecasting Models: Evidence from Turke](https://doi.org/10.1080/1540496x.2017.1419955) |
| 0.599 | 2016 | diğer | Renewable and Sustainable  | [Portfolio optimization under lower partial moments in emerging electrici](https://doi.org/10.1016/j.rser.2016.09.029) |
| 0.585 | 2025 | diğer | 3 SEKTÖR SOSYAL EKONOMİ DE | [Türkiye Enerji Piyasasında Piyasa Takas Fiyatı Tahmini: Makine Öğrenimi ](https://doi.org/10.63556/tisej.2025.1509) |
| 0.585 | 2024 | tahmin | SSRN Electronic Journal | [Electricity Market Price Forecasting Using Deep Lstm Network Optimized w](https://doi.org/10.2139/ssrn.5035223) |
| 0.581 | 2026 | yenilenebilir-etkisi | SSRN Electronic Journal | [Regulation-Induced Financial Curtailment in Electricity Markets: Empiric](https://doi.org/10.2139/ssrn.6478843) |
| 0.580 | 2024 | tahmin | Fırat Üniversitesi Mühendi | [Comparative Analysis of Machine and Deep Learning Methods in Estimating ](https://doi.org/10.35234/fumbd.1473145) |
| 0.577 | 2022 | yenilenebilir-etkisi | Energy Economics | [How do variable renewable energy technologies affect firm-level day-ahea](https://doi.org/10.1016/j.eneco.2022.106169) |
| 0.565 | 2018 | yenilenebilir-etkisi | Energy Research & Social S | [Transformation of the water-energy nexus in Turkey: Re-imagining hydroel](https://doi.org/10.1016/j.erss.2018.04.013) |
| 0.554 | 2018 | oynaklık | RePEc: Research Papers in  | [Market Efficiency and Risk Premium in the Turkish Wholesale Electricity ](https://openalex.org/W2992137884) |
| 0.549 | 2018 | diğer | International Journal of E | [Consideration of Network Constraints in the Turkish Day Ahead Electricit](https://doi.org/10.1016/j.ijepes.2018.04.027) |
| 0.547 | 2011 | politika/tavan | Energy | [Regulation, efficiency and equilibrium: A general equilibrium analysis o](https://doi.org/10.1016/j.energy.2011.03.024) |
| 0.545 | 2017 | tahmin | TURKISH JOURNAL OF ELECTRI | [Probabilistic day-ahead system marginal price forecasting with ANN for t](https://doi.org/10.3906/elk-1612-206) |
| 0.545 | 2020 | yenilenebilir-etkisi | International Journal of E | [THE EFFECTS OF RENEWABLE ENERGY SOURCES ON THE STRUCTURE OF THE TURKISH ](https://doi.org/10.32479/ijeep.8896) |
| 0.543 | 2022 | tahmin | Wind Engineering | [A comprehensive country-based day-ahead wind power generation forecast m](https://doi.org/10.1177/0309524x221078536) |
| 0.542 | 2025 | yenilenebilir-etkisi | International Journal of E | [The influence of hydroelectric power generation and water level variabil](https://doi.org/10.1016/j.ijepes.2025.110865) |
| 0.540 | 2022 | tahmin | Mehmet Akif Ersoy Üniversi | [ELECTRICITY PRICE FORECASTING IN TURKISH DAY-AHEAD MARKET VIA DEEP LEARN](https://doi.org/10.30798/makuiibf.1097686) |
| 0.535 | 2023 | politika/tavan | International series in ma | [An Assessment of Electricity Markets in Turkey: Price Mechanisms, Regula](https://doi.org/10.1007/978-3-031-16620-4_11) |
| 0.534 | 2018 | tahmin | Pressacademia | [A long short term memory application on the Turkish intraday electricity](https://doi.org/10.17261/pressacademia.2018.867) |
| 0.533 | 2025 | tahmin | Computers, materials & con | [Day-Ahead Electricity Price Forecasting Using the XGBoost Algorithm: An ](https://doi.org/10.32604/cmc.2025.068440) |
| 0.530 | 2013 | politika/tavan | Energy Policy | [The effect of power distribution privatization on electricity prices in ](https://doi.org/10.1016/j.enpol.2013.08.090) |
| 0.522 | 2015 | diğer | European Journal of Operat | [On the determination of European day ahead electricity prices: The Turki](https://doi.org/10.1016/j.ejor.2015.02.031) |
| 0.522 | 2017 | diğer | Energy Policy | [Market efficiency assessment under dual pricing rule for the Turkish who](https://doi.org/10.1016/j.enpol.2017.04.024) |
| 0.521 | 2018 | yenilenebilir-etkisi | Journal of Modern Power Sy | [CVaR-based stochastic wind-thermal generation coordination for Turkish e](https://doi.org/10.1007/s40565-018-0492-3) |
| 0.511 | 2022 | oynaklık | Bio-based and Applied Econ | [The Role of Energy on the Price Volatility of Fruits and Vegetables: Evi](https://doi.org/10.36253/bae-10896) |
| 0.508 | 2024 | tahmin | Electric Power Systems Res | [Renewable GenCo bidding strategy using newsvendor-based neural networks:](https://doi.org/10.1016/j.epsr.2024.110301) |
| 0.492 | 2016 | tahmin | ? | [A combined seasonal ARIMA and ANN model for improved results in electric](https://doi.org/10.1109/picmet.2016.7806831) |
| 0.454 | 2019 | politika/tavan | DergiPark (Istanbul Univer | [Comments on Main Factors Affecting Electricity Price Risk in Turkish Ele](https://openalex.org/W3208479772) |
| 0.446 | 2025 | tahmin | ? | [Analysis of Factors Affecting Electricity Prices and Electricity Price F](https://doi.org/10.1109/ubmk67458.2025.11206868) |
| 0.432 | 2022 | diğer | Uluslararası İktisadi ve İ | [ENERGY DERIVATIVES- AN ANALYSIS OF THE TURKISH ELECTRICITY MARKET](https://doi.org/10.18092/ulikidince.930399) |
| 0.430 | 2023 | tahmin | Advances in finance, accou | [Analysis of System Marginal Price in the Turkish Electricity Market](https://doi.org/10.4018/978-1-6684-5976-8.ch013) |
| 0.430 | 2026 | tahmin | SN Business & Economics | [Renewable energy forecast inaccuracies and system marginal price dynamic](https://doi.org/10.1007/s43546-026-01219-0) |
| 0.428 | 2022 | tahmin | ? | [Market-Clearing Price Forecasting Using Keras in Turkish Day-Ahead Elect](https://doi.org/10.1109/gpecom55404.2022.9815603) |
| 0.426 | 2018 | diğer | RePEc: Research Papers in  | [An Adaptive Tabu Search Algorithm for Market Clearing Problem in Turkish](https://doi.org/10.48550/arxiv.1809.10554) |
| 0.404 | 2016 | tahmin | Investment Management and  | [Electricity price forecasting in Turkey with artificial neural network m](https://doi.org/10.21511/imfi.13(3-1).2016.01) |
| 0.404 | 2020 | tahmin | Advances in intelligent sy | [Forecasting the Day-Ahead Prices in Electricity Spot Market of Turkey by](https://doi.org/10.1007/978-3-030-51156-2_122) |
| 0.389 | 2025 | yenilenebilir-etkisi | Balkan Journal of Electric | [Renewable Energy Transition in Türkiye: The Impact of Solar and Wind Bas](https://doi.org/10.17694/bajece.1636476) |
| 0.380 | 2019 | yenilenebilir-etkisi | Akdeniz Üniversitesi İktis | [Türkiye Elektrik Piyasasında Merit Sınıflandırma Etkisinin Test Edilmesi](https://doi.org/10.25294/auiibfd.559394) |
| 0.371 | 2017 | diğer | DergiPark (Istanbul Univer | [Impact of Vertical Integration on Electricity Prices in Turkey](https://openalex.org/W2748945703) |
| 0.360 | 2020 | tahmin | ? | [Electricity Price Prediction Using Encoder-Decoder Recurrent Neural Netw](https://doi.org/10.1109/siu49456.2020.9302070) |
| 0.340 | 2019 | maliyet-geçişi | ? | [Impact of Natural Gas Price on Electricity Price Forecasting in Turkish ](https://doi.org/10.1109/gpecom.2019.8778514) |
| 0.332 | 2015 | tahmin | ? | [Evaluation of price forecast systems for Turkish Electric Market](https://doi.org/10.1109/siu.2015.7129900) |
| 0.321 | 2023 | yenilenebilir-etkisi | Advances in finance, accou | [Renewable Energy Policies and the Future of Energy Storage in Türkiye](https://doi.org/10.4018/979-8-3693-0400-6.ch012) |
| 0.313 | 2025 | yenilenebilir-etkisi | Operations Research Forum | [Power Production by Using the Hydroelectric Power Plants and Cost Effect](https://doi.org/10.1007/s43069-025-00517-x) |
| 0.293 | 2018 | tahmin | OpenMETU (Middle East Tech | [Electricity load and price forecasting of Turkish electricity markets](https://openalex.org/W3115018938) |
| 0.287 | 2015 | politika/tavan | ? | [Turkey’s energy transition milestones and challenges](https://openalex.org/W2260087493) |
| 0.287 | 2011 | diğer | ? | [Distributional Impact Analysis of the Energy Price Reform in Turkey](https://doi.org/10.1596/1813-9450-5831) |
| 0.287 | 2014 | tahmin | İktisat İşletme ve Finans | [Forecasting and Modelling of Electricity Prices by Radial Basis Function](https://doi.org/10.3848/iif.2014.344.4256) |
| 0.285 | 2026 | tahmin | International Advanced Res | [Probabilistic forecasting of short-term electricity prices in the Turkis](https://doi.org/10.35860/iarej.1820591) |
| 0.283 | 2021 | politika/tavan | ? | [Regulation of the Turkish Wholesale Electricity Market: A General Overvi](https://doi.org/10.1007/978-3-030-81720-6_3) |
| 0.280 | 2018 | oynaklık | Marmara Üniversitesi İktis | [MODELLING PRICE DYNAMICS IN TURKISH ELECTRICITY MARKET: LESSONS FROM GAR](https://doi.org/10.14780/muiibd.384221) |
| 0.276 | 2011 | diğer | The Electricity Journal | [Why 2012 Will Be So Important for the Restructured Turkish Electricity M](https://doi.org/10.1016/j.tej.2011.10.018) |
| 0.276 | 2026 | tahmin | Journal of Innovative Scie | [Modeling Turkey’s Hourly Electricity Market Clearing Prices Using Expone](https://doi.org/10.38088/jise.1738364) |
| 0.272 | 2026 | oynaklık | International Journal of E | [Evaluating renewable energy investment flexibility under policy transiti](https://doi.org/10.58559/ijes.1898169) |
| 0.255 | 2018 | diğer | ? | [An Adaptive Tabu Search Algorithm for Market Clearing Problem in Turkish](https://doi.org/10.1109/eem.2018.8469926) |
| 0.244 | 2015 | tahmin | ? | [Short term electricity load forecasting: A case study of electric utilit](https://doi.org/10.1109/sgcf.2015.7354928) |
| 0.233 | 2017 | diğer | DSpace - Isik (Işık Univer | [Turkish electricity sector: a bottom-up approach](https://openalex.org/W2734894508) |

## Konu dağılımı — neyin işlenmiş, neyin işlenmemiş olduğu

| Konu | n | En sağlam örnek |
|---|--:|---|
| tahmin | 43 | [Artificial neural network and SARIMA based models fo](https://doi.org/10.1371/journal.pone.0175915) (2017) |
| yenilenebilir-etkisi | 24 | [The impact of variable renewable energy technologies](https://doi.org/10.1016/j.enpol.2020.112093) (2021) |
| diğer | 18 | [The role of data frequency and method selection in e](https://doi.org/10.1016/j.renene.2021.12.136) (2022) |
| politika/tavan | 9 | [Price spikes, temporary price caps, and welfare effe](https://doi.org/10.1016/j.enpol.2022.112816) (2022) |
| oynaklık | 5 | Market Efficiency and Risk Premium in the Turkish Wh |
| maliyet-geçişi | 2 | [Long‐run relations in European electricity prices](https://doi.org/10.1002/jae.1095) (2009) |
| nedensellik | 1 | [The effects of the Iberian exception mechanism on wh](https://doi.org/10.1080/13504851.2024.2425834) (2024) |

## Her konunun en sağlam 5'i

### tahmin

- **0.833** · 2017 · *PLoS ONE* — [Artificial neural network and SARIMA based models for power load forecasti](https://doi.org/10.1371/journal.pone.0175915)
- **0.828** · 2023 · *Expert Systems with Applications* — [Electricity price estimation using deep learning approaches: An empirical ](https://doi.org/10.1016/j.eswa.2023.120026)
- **0.786** · 2021 · *Applied Energy* — [Data augmentation for time series regression: Applying transformations, au](https://doi.org/10.1016/j.apenergy.2021.117695)
- **0.782** · 2023 · *Journal of Cleaner Production* — [Wind power plants hybridised with solar power: A generation forecast persp](https://doi.org/10.1016/j.jclepro.2023.138793)
- **0.777** · 2021 · *IEEE Access* — [Multi-Horizon Electricity Load and Price Forecasting Using an Interpretabl](https://doi.org/10.1109/access.2021.3086039)

### yenilenebilir-etkisi

- **0.935** · 2021 · *Energy Policy* — [The impact of variable renewable energy technologies on electricity market](https://doi.org/10.1016/j.enpol.2020.112093)
- **0.858** · 2020 · *Energy Policy* — [Variable renewable energy technologies in the Turkish electricity market: ](https://doi.org/10.1016/j.enpol.2020.111660)
- **0.825** · 2019 · *Energy Policy* — [The merit order effect of wind and river type hydroelectricity generation ](https://doi.org/10.1016/j.enpol.2019.07.006)
- **0.788** · 2024 · *Utilities Policy* — [Merit-order of dispatchable and variable renewable energy sources in Turke](https://doi.org/10.1016/j.jup.2024.101758)
- **0.784** · 2022 · *Energies* — [Modelling the Potential Impacts of Nuclear Energy and Renewables in the Tu](https://doi.org/10.3390/en15041392)

### diğer

- **0.747** · 2022 · *Renewable Energy* — [The role of data frequency and method selection in electricity price estim](https://doi.org/10.1016/j.renene.2021.12.136)
- **0.662** · 2021 · *Energies* — [Determination of Price Zones during Transition from Uniform to Zonal Elect](https://doi.org/10.3390/en14041014)
- **0.651** · 2019 · *Manufacturing & Service Operations Managemen* — [Optimizing Day-Ahead Electricity Market Prices: Increasing the Total Surpl](https://doi.org/10.1287/msom.2018.0767)
- **0.631** · 2019 · *Energy Strategy Reviews* — [An analysis of price spikes and deviations in the deregulated Turkish powe](https://doi.org/10.1016/j.esr.2019.100376)
- **0.630** · 2011 · *Renewable and Sustainable Energy Reviews* — [Financial optimization in the Turkish electricity market: Markowitz's mean](https://doi.org/10.1016/j.rser.2011.06.018)

### politika/tavan

- **0.855** · 2022 · *Energy Policy* — [Price spikes, temporary price caps, and welfare effects of regulatory inte](https://doi.org/10.1016/j.enpol.2022.112816)
- **0.684** · 2023 · *Energy Economics* — [Navigating the crisis: Fuel price caps in the Australian national wholesal](https://doi.org/10.1016/j.eneco.2023.107237)
- **0.667** · 2023 · *Journal of the Operational Research Society* — [A dynamic multi-level iterative algorithm for clearing European electricit](https://doi.org/10.1080/01605682.2023.2210182)
- **0.547** · 2011 · *Energy* — [Regulation, efficiency and equilibrium: A general equilibrium analysis of ](https://doi.org/10.1016/j.energy.2011.03.024)
- **0.535** · 2023 · *International series in management science/o* — [An Assessment of Electricity Markets in Turkey: Price Mechanisms, Regulati](https://doi.org/10.1007/978-3-031-16620-4_11)

### oynaklık

- **0.554** · 2018 · *RePEc: Research Papers in Economics* — [Market Efficiency and Risk Premium in the Turkish Wholesale Electricity Ma](https://openalex.org/W2992137884)
- **0.511** · 2022 · *Bio-based and Applied Economics* — [The Role of Energy on the Price Volatility of Fruits and Vegetables: Evide](https://doi.org/10.36253/bae-10896)
- **0.280** · 2018 · *Marmara Üniversitesi İktisadi ve İdari Bilim* — [MODELLING PRICE DYNAMICS IN TURKISH ELECTRICITY MARKET: LESSONS FROM GARCH](https://doi.org/10.14780/muiibd.384221)
- **0.272** · 2026 · *International Journal of Energy Studies* — [Evaluating renewable energy investment flexibility under policy transition](https://doi.org/10.58559/ijes.1898169)
- **0.271** · 2017 · *Journal of Time Series Econometrics* — [Do They Still Matter? – Impact of Fossil Fuels on Electricity Prices in th](https://doi.org/10.1515/jtse-2016-0018)

### maliyet-geçişi

- **0.723** · 2009 · *Journal of Applied Econometrics* — [Long‐run relations in European electricity prices](https://doi.org/10.1002/jae.1095)
- **0.340** · 2019 · *?* — [Impact of Natural Gas Price on Electricity Price Forecasting in Turkish Da](https://doi.org/10.1109/gpecom.2019.8778514)

### nedensellik

- **0.612** · 2024 · *Applied Economics Letters* — [The effects of the Iberian exception mechanism on wholesale electricity pr](https://doi.org/10.1080/13504851.2024.2425834)
