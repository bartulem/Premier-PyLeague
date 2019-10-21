import urllib.request as request
import json
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

seasonsID = {'2007-08': '16', '2008-09': '17', '2009-10': '18', '2010-11': '19', '2011-12': '20', '2012-13': '21', '2013-14': '22', '2014-15': '27', '2015-16': '42', '2016-17': '54', '2017-18': '79', '2018-19': '210'}
playerID = {'David de Gea': '4330', 'Ashley Young': '3062', 'Antonio Valenica': '3285', 'Chris Smalling': '3613', 'Phil Jones': '3820',
            'Jonny Evans': '3156', 'Daley Blind': '4888', 'Marcos Rojo': '10472', 'Luke Shaw': '4608', 'Eric Bailly': '12937',
            'Matteo Darmian': '12854', 'Victor Lindelöf': '5066', 'Diogo Dalot': '24252', 'Rafael da Silva': '3664', 'Rio Ferdinand': '1111',
            'Nemanja Vidic': '2935', 'Patrice Evra': '2934', 'Alexander Büttner': '4537', 'Fabio': '3665', 'Edwin van der Sar': '2070',
            'Gary Neville': '321', 'Wes Brown': '1211', 'John OShea': '1717', 'Gerard Piqué': '2737',
            'Joe Hart': '3143', 'Micah Richards': '2925', 'Pablo Zabaleta': '3654', 'Aleksandar Kolarov': '4141', 'Gaël Clichy': '2413',
            'Vincent Kompany': '3653', 'Joleon Lescott': '2574', 'Martín Demichelis': '4802', 'Matija Nastasic': '4532', 'Bacary Sagna': '3311',
            'Eliaquim Mangala': '5334', 'Claudio Bravo': '5810', 'John Stones': '4505', 'Nicolás Otamendi': '6058', 'Ederson': '12707',
            'Kyle Walker': '3955', 'Danilo': '5328', 'Benjamin Mendy': '5575', 'Aymeric Laporte': '11044', 'Maicon': '4531',
            'Stefan Savic': '4321', 'Wayne Bridge': '1458', 'Jérôme Boateng': '4142', 'Shay Given': '851', 'Javier Garrido': '3427',
            'Sylvinho': '1651', 'Nedum Onuoha': '2728', 'Richard Dunne': '1184', 'Michael Ball': '1185', 'Tal Ben Haim': '2645',
            'Jihai Sun': '2296',
            'Simon Mignolet': '4179', 'Glen Johnson': '2148', 'José Enrique': '3462', 'Aly Cissokho': '4797', 'Jon Flanagan': '4129',
            'Kolo Touré': '2002', 'Daniel Agger': '2916', 'Mamadou Sakho': '4796', 'Martin Skrtel': '3404', 'Javier Manquillo': '4918',
            'Alberto Moreno': '7219', 'Dejan Lovren': '4813', 'James Milner': '2100', 'Loris Karius': '19600', 'Nathaniel Clyne': '4604',
            'Ragnar Klavan': '15608', 'Joël Matip': '5375', 'Trent Alexander-Arnold': '14732', 'Andrew Robertson': '10458',
            'Virgil van Dijk': '5140', 'Joe Gomez': '10651', 'Alisson': '20559', 'Pepe Reina': '2911', 'Sebastián Coates': '4313',
            'Jamie Carragher': '915', 'Andre Wisdom': '4130', 'Martin Kelly': '3644', 'Fábio Aurélio': '3126', 'Sotirios Kyrgiakos': '3900',
            'Emiliano Insúa': '3125', 'Philipp Degen': '3643', 'Álvaro Arbeloa': '3127', 'Andrea Dossena': '3646', 'Sami Hyypiä': '1710',
            'Steve Finnan': '2072', 'John Arne Riise': '2110',
            'Petr Cech': '2651', 'Branislav Ivanovic': '3360', 'César Azpilicueta': '4496', 'Ashley Cole': '1494', 'John Terry': '1353',
            'Gary Cahill': '2620', 'David Luiz': '4100', 'Thibaut Courtois': '4911', 'Filipe Luís': '4915', 'Asmir Begovic': '2537',
            'Kurt Zouma': '5175', 'Marcos Alonso': '4093', 'Willy Caballero': '10466', 'Davide Zappacosta': '25313', 'Antonio Rüdiger': '16801',
            'Andreas Christensen': '4495', 'Kepa': '11373', 'Emerson': '16803', 'Ryan Bertrand': '2886', 'José Bosingwa': '3587',
            'Paulo Ferreira': '2654', 'Yuri Zhirkov': '3860', 'Alex': '2655', 'Juliano Belletti': '3361', 'Ricardo Carvalho': '2653',
            'Wojciech Szczesny': '3543', 'Carl Jenkinson': '4248', 'Nacho Monreal': '4472', 'Kieran Gibbs': '3046', 'Per Mertesacker': '4246',
            'Thomas Vermaelen': '3796', 'Laurent Koscielny': '4030', 'David Ospina': '10420', 'Mathieu Debuchy': '4546', 'Héctor Bellerín': '4474',
            'Gabriel Paulista': '10423', 'Calum Chambers': '4620', 'Rob Holding': '11575', 'Shkodran Mustafi': '3869', 'Sead Kolasinac': '5368',
            'Konstantinos Mavropanos': '32638', 'Bernd Leno': '4985', 'Stephan Lichtsteiner': '5520', 'Sokratis': '5093', 'Johan Djourou': '2609',
            'Emmanuel Eboué': '2612', 'Sébastien Squillaci': '4031', 'Manuel Almunia': '2607', 'Armand Traoré': '2835', 'Sol Campbell': '557',
            'Mikaël Silvestre': '1719', 'Philippe Senderos': '2412',
            'Heurelho Gomes': '3719', 'Danny Rose': '3507', 'Younès Kaboul': '3508', 'Jan Vertonghen': '4666', 'Vlad Chiriches': '4840',
            'Hugo Lloris': '4664', 'Ben Davies': '4408', 'Federico Fazio': '7215', 'Kieran Trippier': '3905', 'Toby Alderweireld': '4916',
            'Eric Dier': '4112', 'Kevin Wimmer': '13813', 'Serge Aurier': '11293', 'Davinson Sánchez': '20224', 'Juan Foyth': '25309',
            'Benoît Assou-Ekotto': '3239', 'William Gallas': '2054', 'Michael Dawson': '2805', 'Ledley King': '1627', 'Alan Hutton': '3506',
            'Sébastien Bassong': '3681',  'Vedran Corluka': '3426', 'Jonathan Woodgate': '1565', 'Gareth Bale': '3513', 'Paul Robinson': '1404',
            'Pascal Chimbonda': '3026', 'Yeong-pyo Lee': '2988'}

injuryCodes = {'David de Gea': '59377', 'Ashley Young': '14086', 'Antonio Valenica': '33544', 'Chris Smalling': '103427', 'Phil Jones': '117996',
               'Jonny Evans': '42412', 'Daley Blind': '12282', 'Marcos Rojo': '93176', 'Luke Shaw': '183288', 'Eric Bailly': '286384',
               'Matteo Darmian': '54906', 'Victor Lindelöf': '184573', 'Diogo Dalot': '357147', 'Rafael da Silva': '61892', 'Rio Ferdinand': '3235',
               'Nemanja Vidic': '19726', 'Patrice Evra': '5285', 'Alexander Büttner': '38003', 'Fabio': '61891', 'Edwin van der Sar': '3516',
               'Gary Neville': '3403', 'Wes Brown': '3405', 'John OShea': '3540', 'Gerard Piqué': '18944',
               'Joe Hart': '40204', 'Micah Richards': '32617', 'Pablo Zabaleta': '20007', 'Aleksandar Kolarov': '46156', 'Gaël Clichy': '7449',
               'Vincent Kompany': '9594', 'Joleon Lescott': '4241', 'Martín Demichelis': '2963', 'Matija Nastasic': '143559', 'Bacary Sagna': '26764',
               'Eliaquim Mangala': '90681', 'Claudio Bravo': '40423', 'John Stones': '186590', 'Nicolás Otamendi': '54781', 'Ederson': '238223',
               'Kyle Walker': '95424', 'Danilo': '145707', 'Benjamin Mendy': '157495', 'Aymeric Laporte': '176553', 'Maicon': '18301',
               'Stefan Savic': '107010', 'Wayne Bridge': '3682', 'Jérôme Boateng': '26485', 'Shay Given': '3146', 'Javier Garrido': '23218',
               'Sylvinho': '3359', 'Nedum Onuoha': '28810', 'Richard Dunne': '3807', 'Michael Ball': '9423', 'Tal Ben Haim': '20714',
               'Jihai Sun': '3806',
               'Simon Mignolet': '50219', 'Glen Johnson': '3881', 'José Enrique': '35571', 'Aly Cissokho': '57515', 'Jon Flanagan': '145922',
               'Kolo Touré': '3202', 'Daniel Agger': '22832', 'Mamadou Sakho': '47713', 'Martin Skrtel': '24180', 'Javier Manquillo': '162029',
               'Alberto Moreno': '207917', 'Dejan Lovren': '37838', 'James Milner': '3333', 'Loris Karius': '85864', 'Nathaniel Clyne': '85177',
               'Ragnar Klavan': '26669', 'Joël Matip': '82105', 'Trent Alexander-Arnold': '314353', 'Andrew Robertson': '234803',
               'Virgil van Dijk': '139208', 'Joe Gomez': '256178', 'Alisson': '105470', 'Pepe Reina': '7825', 'Sebastián Coates': '102427',
               'Jamie Carragher': '3597', 'Andre Wisdom': '128912', 'Martin Kelly': '78959', 'Fábio Aurélio': '7570', 'Sotirios Kyrgiakos': '9706',
               'Emiliano Insúa': '45599', 'Philipp Degen': '2895', 'Álvaro Arbeloa': '28260', 'Andrea Dossena': '21863', 'Sami Hyypiä': '3470',
               'Steve Finnan': '4038', 'John Arne Riise': '3220',
               'Petr Cech': '5658', 'Branislav Ivanovic': '36827', 'César Azpilicueta': '57500', 'Ashley Cole': '3182', 'John Terry': '3160',
               'Gary Cahill': '27511', 'David Luiz': '46741', 'Thibaut Courtois': '108390', 'Filipe Luís': '21725', 'Asmir Begovic': '33873',
               'Kurt Zouma': '157509', 'Marcos Alonso': '112515', 'Willy Caballero': '19948', 'Davide Zappacosta': '173859', 'Antonio Rüdiger': '86202',
               'Andreas Christensen': '196948', 'Kepa': '192279', 'Emerson': '181778', 'Ryan Bertrand': '40611', 'José Bosingwa': '9813',
               'Paulo Ferreira': '9832', 'Yuri Zhirkov': '16760', 'Alex': '15420', 'Juliano Belletti': '7829', 'Ricardo Carvalho': '9828',
               'Wojciech Szczesny': '44058', 'Carl Jenkinson': '126321', 'Nacho Monreal': '43003', 'Kieran Gibbs': '44792', 'Per Mertesacker': '6710',
               'Thomas Vermaelen': '15904', 'Laurent Koscielny': '76277', 'David Ospina': '73396', 'Mathieu Debuchy': '27306', 'Héctor Bellerín': '191217',
               'Gabriel Paulista': '149498', 'Calum Chambers': '215118', 'Rob Holding': '253341', 'Shkodran Mustafi': '88590', 'Sead Kolasinac': '94005',
               'Konstantinos Mavropanos': '415912', 'Bernd Leno': '72476', 'Stephan Lichtsteiner': '2865', 'Sokratis': '34322', 'Johan Djourou': '34561',
               'Emmanuel Eboué': '13058', 'Sébastien Squillaci': '5293', 'Manuel Almunia': '16621', 'Armand Traoré': '33783', 'Sol Campbell': '3198',
               'Mikaël Silvestre': '3393', 'Philippe Senderos': '4277',
               'Heurelho Gomes': '19059', 'Danny Rose': '50174', 'Younès Kaboul': '27114', 'Jan Vertonghen': '43250', 'Vlad Chiriches': '123261',
               'Hugo Lloris': '17965', 'Ben Davies': '192765', 'Federico Fazio': '45314', 'Kieran Trippier': '95810', 'Toby Alderweireld': '42710',
               'Eric Dier': '175722', 'Kevin Wimmer': '122675', 'Serge Aurier': '127032', 'Davinson Sánchez': '341429', 'Juan Foyth': '480763',
               'Benoît Assou-Ekotto': '18310', 'William Gallas': '3156', 'Michael Dawson': '9988', 'Ledley King': '3360', 'Alan Hutton': '9619',
               'Sébastien Bassong': '33951', 'Vedran Corluka': '34393', 'Jonathan Woodgate': '3224', 'Gareth Bale': '39381', 'Paul Robinson': '3630',
               'Pascal Chimbonda': '18875', 'Yeong-pyo Lee': '6156'}

playersBYseasons = {'2007-08': {'MU': {'GK': ['Edwin van der Sar'], 'RB': ['Wes Brown', 'John OShea'], 'LB': ['Patrice Evra'], 'CB': ['Rio Ferdinand', 'Nemanja Vidic', 'Gerard Piqué', 'Mikaël Silvestre']},
                                'MC': {'GK': ['Joe Hart'], 'RB': ['Micah Richards'], 'LB': ['Michael Ball', 'Javier Garrido'], 'CB': ['Nedum Onuoha', 'Jihai Sun', 'Richard Dunne', 'Vedran Corluka']},
                                'LFC': {'GK': ['Pepe Reina'], 'RB': ['Álvaro Arbeloa', 'Steve Finnan'], 'LB': ['John Arne Riise', 'Fábio Aurélio', 'Emiliano Insúa'], 'CB': ['Daniel Agger', 'Martin Skrtel', 'Jamie Carragher', 'Sami Hyypiä']},
                                'CFC': {'GK': ['Petr Cech'], 'RB': ['Paulo Ferreira', 'Juliano Belletti'], 'LB': ['Ashley Cole', 'Wayne Bridge'], 'CB': ['John Terry', 'Alex', 'Ricardo Carvalho', 'Tal Ben Haim']},
                                'AFC': {'GK': ['Manuel Almunia'], 'RB': ['Bacary Sagna', 'Emmanuel Eboué'], 'LB': ['Gaël Clichy', 'Armand Traoré'], 'CB': ['William Gallas', 'Kolo Touré', 'Johan Djourou', 'Philippe Senderos']},
                                'TH': {'GK': ['Paul Robinson'], 'RB': ['Pascal Chimbonda', 'Alan Hutton'], 'LB': ['Yeong-pyo Lee', 'Benoît Assou-Ekotto', 'Gareth Bale'], 'CB': ['Younès Kaboul', 'Ledley King', 'Michael Dawson', 'Jonathan Woodgate']}},
                    '2008-09': {'MU': {'GK': ['Edwin van der Sar'], 'RB': ['Rafael da Silva', 'Gary Neville', 'John OShea'], 'LB': ['Patrice Evra'], 'CB': ['Rio Ferdinand', 'Nemanja Vidic', 'Jonny Evans', 'Wes Brown']},
                                'MC': {'GK': ['Joe Hart'], 'RB': ['Micah Richards', 'Pablo Zabaleta'], 'LB': ['Michael Ball', 'Javier Garrido', 'Wayne Bridge'], 'CB': ['Vincent Kompany', 'Nedum Onuoha', 'Richard Dunne', 'Tal Ben Haim']},
                                'LFC': {'GK': ['Pepe Reina'], 'RB': ['Álvaro Arbeloa'], 'LB': ['Fábio Aurélio', 'Emiliano Insúa', 'Andrea Dossena'], 'CB': ['Daniel Agger', 'Martin Skrtel', 'Jamie Carragher', 'Sami Hyypiä']},
                                'CFC': {'GK': ['Petr Cech'], 'RB': ['Branislav Ivanovic', 'José Bosingwa', 'Paulo Ferreira', 'Juliano Belletti'], 'LB': ['Ashley Cole'], 'CB': ['John Terry', 'Alex', 'Ricardo Carvalho']},
                                'AFC': {'GK': ['Manuel Almunia'], 'RB': ['Bacary Sagna', 'Emmanuel Eboué'], 'LB': ['Kieran Gibbs', 'Gaël Clichy'], 'CB': ['William Gallas', 'Kolo Touré', 'Johan Djourou', 'Mikaël Silvestre']},
                                'TH': {'GK': ['Heurelho Gomes'], 'RB': ['Alan Hutton'], 'LB': ['Benoît Assou-Ekotto', 'Gareth Bale'], 'CB': ['Ledley King', 'Vedran Corluka', 'Michael Dawson', 'Jonathan Woodgate']}},
                    '2009-10': {'MU': {'GK': ['Edwin van der Sar'], 'RB': ['Rafael da Silva', 'Fabio', 'Gary Neville', 'John OShea'], 'LB': ['Patrice Evra'], 'CB': ['Rio Ferdinand', 'Nemanja Vidic', 'Jonny Evans', 'Wes Brown']},
                                'MC': {'GK': ['Shay Given'], 'RB': ['Micah Richards', 'Pablo Zabaleta'], 'LB': ['Javier Garrido', 'Wayne Bridge', 'Sylvinho'], 'CB': ['Vincent Kompany', 'Joleon Lescott', 'Nedum Onuoha', 'Kolo Touré', 'Richard Dunne']},
                                'LFC': {'GK': ['Pepe Reina'], 'RB': ['Glen Johnson', 'Philipp Degen'], 'LB': ['Fábio Aurélio', 'Emiliano Insúa'], 'CB': ['Daniel Agger', 'Sotirios Kyrgiakos', 'Martin Skrtel', 'Jamie Carragher']},
                                'CFC': {'GK': ['Petr Cech'], 'RB': ['Branislav Ivanovic', 'José Bosingwa', 'Paulo Ferreira', 'Juliano Belletti'], 'LB': ['Ashley Cole', 'Yuri Zhirkov'], 'CB': ['John Terry', 'Alex', 'Ricardo Carvalho']},
                                'AFC': {'GK': ['Manuel Almunia'], 'RB': ['Bacary Sagna', 'Emmanuel Eboué'], 'LB': ['Kieran Gibbs', 'Gaël Clichy', 'Armand Traoré'], 'CB': ['William Gallas', 'Thomas Vermaelen', 'Sol Campbell', 'Johan Djourou', 'Mikaël Silvestre']},
                                'TH': {'GK': ['Heurelho Gomes'], 'RB': ['Kyle Walker', 'Alan Hutton'], 'LB': ['Danny Rose', 'Benoît Assou-Ekotto'], 'CB': ['Younès Kaboul', 'Ledley King', 'Vedran Corluka', 'Michael Dawson', 'Sébastien Bassong', 'Jonathan Woodgate']}},
                    '2010-11': {'MU': {'GK': ['Edwin van der Sar'], 'RB': ['Rafael da Silva', 'Fabio', 'Gary Neville', 'John OShea'], 'LB': ['Patrice Evra'], 'CB': ['Rio Ferdinand', 'Nemanja Vidic', 'Jonny Evans', 'Wes Brown', 'Chris Smalling']},
                                'MC': {'GK': ['Joe Hart'], 'RB': ['Micah Richards', 'Pablo Zabaleta'], 'LB': ['Aleksandar Kolarov', 'Wayne Bridge'], 'CB': ['Vincent Kompany', 'Joleon Lescott', 'Jérôme Boateng', 'Kolo Touré']},
                                'LFC': {'GK': ['Pepe Reina'], 'RB': ['Glen Johnson'], 'LB': ['Fábio Aurélio', 'Martin Kelly', 'Jon Flanagan'], 'CB': ['Daniel Agger', 'Sotirios Kyrgiakos', 'Martin Skrtel', 'Jamie Carragher']},
                                'CFC': {'GK': ['Petr Cech'], 'RB': ['Branislav Ivanovic', 'José Bosingwa', 'Paulo Ferreira'], 'LB': ['Ashley Cole', 'Yuri Zhirkov'], 'CB': ['John Terry', 'David Luiz', 'Alex']},
                                'AFC': {'GK': ['Wojciech Szczesny'], 'RB': ['Bacary Sagna', 'Emmanuel Eboué'], 'LB': ['Kieran Gibbs', 'Gaël Clichy'], 'CB': ['Thomas Vermaelen', 'Laurent Koscielny', 'Johan Djourou', 'Sébastien Squillaci']},
                                'TH': {'GK': ['Heurelho Gomes'], 'RB': ['Kyle Walker', 'Alan Hutton'], 'LB': ['Danny Rose', 'Benoît Assou-Ekotto'], 'CB': ['Younès Kaboul', 'Ledley King', 'William Gallas', 'Michael Dawson', 'Sébastien Bassong']}},
                    '2011-12': {'MU': {'GK': ['David de Gea'], 'RB': ['Rafael da Silva', 'Fabio'], 'LB': ['Patrice Evra'], 'CB': ['Rio Ferdinand', 'Nemanja Vidic', 'Jonny Evans', 'Chris Smalling', 'Phil Jones']},
                                'MC': {'GK': ['Joe Hart'], 'RB': ['Micah Richards', 'Pablo Zabaleta'], 'LB': ['Aleksandar Kolarov', 'Gaël Clichy'], 'CB': ['Vincent Kompany', 'Joleon Lescott', 'Stefan Savic', 'Kolo Touré']},
                                'LFC': {'GK': ['Pepe Reina'], 'RB': ['Glen Johnson'], 'LB': ['José Enrique', 'Martin Kelly', 'Jon Flanagan'], 'CB': ['Daniel Agger', 'Sebastián Coates', 'Martin Skrtel', 'Jamie Carragher']},
                                'CFC': {'GK': ['Petr Cech'], 'RB': ['Branislav Ivanovic', 'José Bosingwa', 'Paulo Ferreira'], 'LB': ['Ashley Cole', 'Ryan Bertrand'], 'CB': ['John Terry', 'Gary Cahill', 'David Luiz']},
                                'AFC': {'GK': ['Wojciech Szczesny'], 'RB': ['Bacary Sagna', 'Carl Jenkinson'], 'LB': ['Kieran Gibbs'], 'CB': ['Per Mertesacker', 'Thomas Vermaelen', 'Laurent Koscielny', 'Johan Djourou']},
                                'TH': {'GK': ['Heurelho Gomes'], 'RB': ['Kyle Walker'], 'LB': ['Benoît Assou-Ekotto'], 'CB': ['Younès Kaboul', 'Ledley King', 'William Gallas', 'Michael Dawson']}},
                    '2012-13': {'MU': {'GK': ['David de Gea'], 'RB': ['Rafael da Silva', 'Antonio Valenica'], 'LB': ['Patrice Evra', 'Alexander Büttner'], 'CB': ['Rio Ferdinand', 'Nemanja Vidic', 'Jonny Evans', 'Chris Smalling', 'Phil Jones']},
                                'MC': {'GK': ['Joe Hart'], 'RB': ['Micah Richards', 'Maicon', 'Pablo Zabaleta'], 'LB': ['Aleksandar Kolarov', 'Gaël Clichy'], 'CB': ['Vincent Kompany', 'Joleon Lescott', 'Matija Nastasic', 'Kolo Touré']},
                                'LFC': {'GK': ['Pepe Reina'], 'RB': ['Glen Johnson'], 'LB': ['José Enrique', 'Martin Kelly'], 'CB': ['Daniel Agger', 'Sebastián Coates', 'Martin Skrtel', 'Jamie Carragher', 'Andre Wisdom']},
                                'CFC': {'GK': ['Petr Cech'], 'RB': ['Branislav Ivanovic', 'César Azpilicueta'], 'LB': ['Ashley Cole', 'Ryan Bertrand'], 'CB': ['John Terry', 'Gary Cahill', 'David Luiz']},
                                'AFC': {'GK': ['Wojciech Szczesny'], 'RB': ['Bacary Sagna', 'Carl Jenkinson'], 'LB': ['Nacho Monreal', 'Kieran Gibbs'], 'CB': ['Per Mertesacker', 'Thomas Vermaelen', 'Laurent Koscielny']},
                                'TH': {'GK': ['Heurelho Gomes'], 'RB': ['Kyle Walker'], 'LB': ['Danny Rose', 'Benoît Assou-Ekotto'], 'CB': ['Younès Kaboul', 'Jan Vertonghen', 'William Gallas', 'Michael Dawson']}},
                    '2013-14': {'MU': {'GK': ['David de Gea'], 'RB': ['Rafael da Silva', 'Antonio Valenica'], 'LB': ['Patrice Evra', 'Alexander Büttner'], 'CB': ['Rio Ferdinand', 'Nemanja Vidic', 'Jonny Evans', 'Chris Smalling', 'Phil Jones']},
                                'MC': {'GK': ['Joe Hart'], 'RB': ['Pablo Zabaleta'], 'LB': ['Aleksandar Kolarov', 'Gaël Clichy'], 'CB': ['Vincent Kompany', 'Joleon Lescott', 'Martín Demichelis', 'Matija Nastasic']},
                                'LFC': {'GK': ['Simon Mignolet'], 'RB': ['Glen Johnson'], 'LB': ['José Enrique', 'Aly Cissokho', 'Jon Flanagan'], 'CB': ['Kolo Touré', 'Daniel Agger', 'Mamadou Sakho', 'Martin Skrtel']},
                                'CFC': {'GK': ['Petr Cech'], 'RB': ['Branislav Ivanovic', 'César Azpilicueta'], 'LB': ['Ashley Cole'], 'CB': ['John Terry', 'Gary Cahill', 'David Luiz']},
                                'AFC': {'GK': ['Wojciech Szczesny'], 'RB': ['Bacary Sagna', 'Carl Jenkinson'], 'LB': ['Nacho Monreal', 'Kieran Gibbs'], 'CB': ['Per Mertesacker', 'Thomas Vermaelen', 'Laurent Koscielny']},
                                'TH': {'GK': ['Hugo Lloris'], 'RB': ['Kyle Walker'], 'LB': ['Danny Rose'], 'CB': ['Younès Kaboul', 'Jan Vertonghen', 'Vlad Chiriches']}},
                    '2014-15': {'MU': {'GK': ['David de Gea'], 'RB': ['Rafael da Silva', 'Antonio Valenica'], 'LB': ['Ashley Young', 'Luke Shaw'], 'CB': ['Jonny Evans', 'Chris Smalling', 'Phil Jones', 'Marcos Rojo', 'Daley Blind']},
                                'MC': {'GK': ['Joe Hart'], 'RB': ['Pablo Zabaleta', 'Bacary Sagna'], 'LB': ['Aleksandar Kolarov', 'Gaël Clichy'], 'CB': ['Vincent Kompany', 'Martín Demichelis', 'Matija Nastasic', 'Eliaquim Mangala']},
                                'LFC': {'GK': ['Simon Mignolet'], 'RB': ['Glen Johnson', 'Javier Manquillo'], 'LB': ['Alberto Moreno'], 'CB': ['Dejan Lovren', 'Mamadou Sakho', 'Martin Skrtel']},
                                'CFC': {'GK': ['Thibaut Courtois', 'Petr Cech'], 'RB': ['Branislav Ivanovic'], 'LB': ['César Azpilicueta', 'Filipe Luís'], 'CB': ['John Terry', 'Gary Cahill']},
                                'AFC': {'GK': ['Wojciech Szczesny'], 'RB': ['Mathieu Debuchy', 'Héctor Bellerín'], 'LB': ['Nacho Monreal', 'Kieran Gibbs'], 'CB': ['Per Mertesacker', 'Gabriel Paulista', 'Laurent Koscielny', 'Calum Chambers']},
                                'TH': {'GK': ['Hugo Lloris'], 'RB': ['Kyle Walker'], 'LB': ['Danny Rose', 'Ben Davies'], 'CB': ['Younès Kaboul', 'Jan Vertonghen', 'Vlad Chiriches', 'Federico Fazio']}},
                    '2015-16': {'MU': {'GK': ['David de Gea'], 'RB': ['Matteo Darmian', 'Antonio Valenica'], 'LB': ['Ashley Young', 'Luke Shaw'], 'CB': ['Chris Smalling', 'Phil Jones', 'Marcos Rojo', 'Daley Blind']},
                                'MC': {'GK': ['Joe Hart'], 'RB': ['Pablo Zabaleta', 'Bacary Sagna'], 'LB': ['Aleksandar Kolarov', 'Gaël Clichy'], 'CB': ['Vincent Kompany', 'Martín Demichelis', 'Eliaquim Mangala']},
                                'LFC': {'GK': ['Simon Mignolet'], 'RB': ['James Milner'], 'LB': ['Alberto Moreno'], 'CB': ['Dejan Lovren', 'Mamadou Sakho', 'Martin Skrtel']},
                                'CFC': {'GK': ['Thibaut Courtois', 'Asmir Begovic'], 'RB': ['Branislav Ivanovic'], 'LB': ['César Azpilicueta'], 'CB': ['John Terry', 'Gary Cahill', 'Kurt Zouma']},
                                'AFC': {'GK': ['Petr Cech'], 'RB': ['Héctor Bellerín'], 'LB': ['Nacho Monreal', 'Kieran Gibbs'], 'CB': ['Per Mertesacker', 'Gabriel Paulista', 'Laurent Koscielny', 'Calum Chambers']},
                                'TH': {'GK': ['Hugo Lloris'], 'RB': ['Kyle Walker', 'Kieran Trippier'], 'LB': ['Danny Rose', 'Ben Davies'], 'CB': ['Toby Alderweireld', 'Jan Vertonghen', 'Eric Dier', 'Kevin Wimmer']}},
                    '2016-17': {'MU': {'GK': ['David de Gea'], 'RB': ['Matteo Darmian', 'Antonio Valenica'], 'LB': ['Ashley Young', 'Luke Shaw'], 'CB': ['Chris Smalling', 'Phil Jones', 'Marcos Rojo', 'Daley Blind', 'Eric Bailly']},
                                'MC': {'GK': ['Claudio Bravo'], 'RB': ['Pablo Zabaleta', 'Bacary Sagna'], 'LB': ['Aleksandar Kolarov', 'Gaël Clichy'], 'CB': ['Vincent Kompany', 'John Stones', 'Nicolás Otamendi']},
                                'LFC': {'GK': ['Simon Mignolet', 'Loris Karius'], 'RB': ['Nathaniel Clyne'], 'LB': ['James Milner'], 'CB': ['Dejan Lovren', 'Ragnar Klavan', 'Joël Matip']},
                                'CFC': {'GK': ['Thibaut Courtois', 'Asmir Begovic'], 'RB': ['César Azpilicueta'], 'LB': ['Marcos Alonso'], 'CB': ['John Terry', 'Gary Cahill', 'Kurt Zouma', 'David Luiz']},
                                'AFC': {'GK': ['Petr Cech'], 'RB': ['Héctor Bellerín'], 'LB': ['Nacho Monreal', 'Kieran Gibbs'], 'CB': ['Per Mertesacker', 'Gabriel Paulista', 'Laurent Koscielny', 'Rob Holding', 'Shkodran Mustafi']},
                                'TH': {'GK': ['Hugo Lloris'], 'RB': ['Kyle Walker', 'Kieran Trippier'], 'LB': ['Danny Rose', 'Ben Davies'], 'CB': ['Toby Alderweireld', 'Jan Vertonghen', 'Kevin Wimmer']}},
                    '2017-18': {'MU': {'GK': ['David de Gea'], 'RB': ['Ashley Young', 'Matteo Darmian', 'Antonio Valenica'], 'LB': ['Luke Shaw'], 'CB': ['Chris Smalling', 'Phil Jones', 'Marcos Rojo', 'Daley Blind', 'Eric Bailly', 'Victor Lindelöf']},
                                'MC': {'GK': ['Ederson'], 'RB': ['Kyle Walker', 'Danilo'], 'LB': ['Benjamin Mendy'], 'CB': ['Vincent Kompany', 'John Stones', 'Nicolás Otamendi', 'Aymeric Laporte', 'Eliaquim Mangala']},
                                'LFC': {'GK': ['Simon Mignolet', 'Loris Karius'], 'RB': ['Trent Alexander-Arnold'], 'LB': ['Alberto Moreno', 'Andrew Robertson'], 'CB': ['Virgil van Dijk', 'Dejan Lovren', 'Joe Gomez', 'Ragnar Klavan', 'Joël Matip']},
                                'CFC': {'GK': ['Thibaut Courtois', 'Willy Caballero'], 'RB': ['Davide Zappacosta', 'César Azpilicueta'], 'LB': ['Marcos Alonso'], 'CB': ['Antonio Rüdiger', 'Gary Cahill', 'Andreas Christensen', 'David Luiz']},
                                'AFC': {'GK': ['Petr Cech'], 'RB': ['Héctor Bellerín'], 'LB': ['Nacho Monreal', 'Sead Kolasinac'], 'CB': ['Per Mertesacker', 'Laurent Koscielny', 'Rob Holding', 'Shkodran Mustafi', 'Calum Chambers', 'Konstantinos Mavropanos']},
                                'TH': {'GK': ['Hugo Lloris'], 'RB': ['Kieran Trippier', 'Serge Aurier'], 'LB': ['Danny Rose', 'Ben Davies'], 'CB': ['Toby Alderweireld', 'Jan Vertonghen', 'Davinson Sánchez']}},
                    '2018-19': {'MU': {'GK': ['David de Gea'], 'RB': ['Ashley Young', 'Matteo Darmian', 'Diogo Dalot', 'Antonio Valenica'], 'LB': ['Luke Shaw'], 'CB': ['Chris Smalling', 'Phil Jones', 'Marcos Rojo', 'Eric Bailly', 'Victor Lindelöf']},
                                'MC': {'GK': ['Ederson'], 'RB': ['Kyle Walker', 'Danilo'], 'LB': ['Benjamin Mendy'], 'CB': ['Vincent Kompany', 'John Stones', 'Nicolás Otamendi', 'Aymeric Laporte', 'Eliaquim Mangala']},
                                'LFC': {'GK': ['Alisson'], 'RB': ['Trent Alexander-Arnold'], 'LB': ['Andrew Robertson'], 'CB': ['Virgil van Dijk', 'Dejan Lovren', 'Joe Gomez', 'Joël Matip']},
                                'CFC': {'GK': ['Kepa', 'Willy Caballero'], 'RB': ['César Azpilicueta'], 'LB': ['Marcos Alonso', 'Emerson'], 'CB': ['Antonio Rüdiger', 'Andreas Christensen', 'David Luiz']},
                                'AFC': {'GK': ['Bernd Leno'], 'RB': ['Héctor Bellerín', 'Stephan Lichtsteiner'], 'LB': ['Nacho Monreal', 'Sead Kolasinac'], 'CB': ['Laurent Koscielny', 'Rob Holding', 'Shkodran Mustafi', 'Konstantinos Mavropanos', 'Sokratis']},
                                'TH': {'GK': ['Hugo Lloris'], 'RB': ['Kieran Trippier', 'Serge Aurier'], 'LB': ['Danny Rose', 'Ben Davies'], 'CB': ['Toby Alderweireld', 'Jan Vertonghen', 'Davinson Sánchez', 'Juan Foyth']}}}


def getIndex(spicificseason):
    playersLst = []
    for aclub in playersBYseasons[spicificseason].keys():
        for aposition in playersBYseasons[spicificseason][aclub].keys():
            for aplayer in playersBYseasons[spicificseason][aclub][aposition]:
                playersLst.append(aplayer)
    return playersLst


def getColumns(spicificseason):
    allVars = []
    for aclub in playersBYseasons[spicificseason].keys():
        for aposition in playersBYseasons[spicificseason][aclub].keys():
            for aplayer in playersBYseasons[spicificseason][aclub][aposition]:
                urlTemp = 'https://footballapi.pulselive.com/football/stats/player/{}?comps=1&compSeasons={}'.format(playerID[aplayer], seasonsID[spicificseason])
                with request.urlopen(urlTemp) as response:
                    if(response.getcode() == 200):
                        dataPlayer = json.loads(response.read())
                    else:
                        print('An error occurred for {} in season {}'.format(aplayer, spicificseason))
                for avariable in dataPlayer['stats']:
                    if(avariable['name'] not in allVars):
                        allVars.append(avariable['name'])
    return sorted(allVars)


def getInjuries(aplayer, theseason):
    aplayerCleaned = aplayer.lower().replace(' ', '-').replace('ö', 'o').replace('ë', 'e').replace('á', 'a').replace('í', 'i').replace('é', 'e').replace('î', 'i').replace('ü', 'u').replace('è', 'e').replace('ô', 'o').replace('ú', 'u').replace('Á', 'A').replace('ä', 'a')
    injuryurl = 'https://www.transfermarkt.com/{}/verletzungen/spieler/{}'.format(aplayerCleaned, injuryCodes[aplayer])
    response = requests.get(injuryurl, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')
    numOfPages = [np.int(x.text) for x in soup.findAll('li', {'class': 'page'})]
    soup.decompose()
    if(not numOfPages):
        numOfPages = [1]

    daysInjured = []
    gamesMissed = []
    seasonOfInjury = []
    for apage in numOfPages:
        injuryurltemp = '{}/ajax/yw1/page/{}'.format(injuryurl, apage)
        responsetemp = requests.get(injuryurltemp, headers={'User-Agent': 'Mozilla/5.0'})
        souptemp = BeautifulSoup(responsetemp.text, 'html.parser')

        for aday in [np.float(x.text.strip(' days')) for x in souptemp.findAll('td', {'class': 'rechts'})[::2]]:
            daysInjured.append(aday)

        for agame in [np.float(x.text) if x.text.isdigit() else 0 for x in souptemp.findAll('td', {'class': 'rechts'})[1::2]]:
            gamesMissed.append(agame)

        for oneseason in ['{}{}'.format('20', x.text.replace('/', '-')) for x in souptemp.findAll('td', {'class': 'zentriert'})[::3]]:
            seasonOfInjury.append(oneseason)

        souptemp.decompose()

    daysInjured = np.array(daysInjured)
    gamesMissed = np.array(gamesMissed)

    if(theseason not in seasonOfInjury):
        return (0, 0)
    else:
        injuryboolean = [True if elem == theseason else False for elem in seasonOfInjury]
        seasonDaysInjured = np.nansum(daysInjured[injuryboolean])
        seasonGamesMissed = np.nansum(gamesMissed[injuryboolean])
        return seasonDaysInjured, seasonGamesMissed


def dataParser():
    dataDict = {}
    for aseason in tqdm(playersBYseasons.keys()):
        dfTemp = pd.DataFrame(index=getIndex(aseason), columns=['club', 'position', 'days_injured', 'games_missed'] + getColumns(aseason), dtype=np.float)
        for aclub in playersBYseasons[aseason].keys():
            for aposition in playersBYseasons[aseason][aclub].keys():
                for aplayer in playersBYseasons[aseason][aclub][aposition]:
                    urlTemp = 'https://footballapi.pulselive.com/football/stats/player/{}?comps=1&compSeasons={}'.format(playerID[aplayer], seasonsID[aseason])
                    with request.urlopen(urlTemp) as response:
                        if(response.getcode() == 200):
                            dataPlayer = json.loads(response.read())
                        else:
                            print('An error occurred for {} in season {}'.format(aplayer, aseason))
                    days_injured, games_missed = getInjuries(aplayer, aseason)
                    inames = ['club', 'position', 'days_injured', 'games_missed'] + [avariable['name'] for avariable in dataPlayer['stats']]
                    ivalues = [aclub, aposition, days_injured, games_missed] + [avariable['value'] for avariable in dataPlayer['stats']]
                    idict = {key: value for key, value in zip(inames, ivalues)}
                    for acolumn in dfTemp.columns:
                        if(acolumn in idict.keys()):
                            dfTemp.loc[aplayer, acolumn] = idict[acolumn]
                        else:
                            dfTemp.loc[aplayer, acolumn] = 0.
        dataDict[aseason] = dfTemp

    pickle.dump(dataDict, open('premierleague_data', 'wb'))


dataParser()
