# app.py

from flask import Flask, request, jsonify
import numpy as np
import torch
from torch import nn
import joblib

app = Flask(__name__)

# Your model and other setup code here
class SimpleNet(nn.Module):
    def __init__(self, input_size=819):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),  # Modify the input size here
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        yhat = self.layers(x)
        # yhat = self(x)
        return yhat
    
    def predict(self, x):
        # Pass the input through the neural network
        with torch.no_grad():  # Ensure that no gradients are computed during prediction
            yhat = self(x)
        return yhat
    
model = SimpleNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

all_airlines = [
  'Airline_Air Wisconsin Airlines Corp',
  'Airline_Alaska Airlines Inc.',
  'Airline_Allegiant Air',
  'Airline_American Airlines Inc.',
  'Airline_Capital Cargo International',
  'Airline_Comair Inc.',
  'Airline_Commutair Aka Champlain Enterprises, Inc.',
  'Airline_Delta Air Lines Inc.',
  'Airline_Endeavor Air Inc.',
  'Airline_Envoy Air',
  'Airline_Frontier Airlines Inc.',
  'Airline_GoJet Airlines, LLC d/b/a United Express',
  'Airline_Hawaiian Airlines Inc.',
  'Airline_Horizon Air',
  'Airline_JetBlue Airways',
  'Airline_Mesa Airlines Inc.',
  'Airline_Republic Airlines',
  'Airline_SkyWest Airlines Inc.',
  'Airline_Southwest Airlines Co.',
  'Airline_Spirit Air Lines',
  'Airline_United Air Lines Inc.'
];

all_months = [
    'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7'
];

all_daysofmonth = [
    'DayofMonth_1', 'DayofMonth_2', 'DayofMonth_3', 'DayofMonth_4', 'DayofMonth_5', 'DayofMonth_6', 'DayofMonth_7', 'DayofMonth_8', 'DayofMonth_9', 'DayofMonth_10', 'DayofMonth_11', 'DayofMonth_12', 'DayofMonth_13', 'DayofMonth_14', 'DayofMonth_15', 'DayofMonth_16', 'DayofMonth_17', 'DayofMonth_18', 'DayofMonth_19', 'DayofMonth_20', 'DayofMonth_21', 'DayofMonth_22', 'DayofMonth_23', 'DayofMonth_24', 'DayofMonth_25', 'DayofMonth_26', 'DayofMonth_27', 'DayofMonth_28', 'DayofMonth_29', 'DayofMonth_30', 'DayofMonth_31'
];

all_daysofweek = [
    'DayOfWeek_1', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7'
];

all_originairportids = [
    'OriginAirportID_10135', 'OriginAirportID_10136', 'OriginAirportID_10140', 'OriginAirportID_10141', 'OriginAirportID_10146', 'OriginAirportID_10154', 'OriginAirportID_10155', 'OriginAirportID_10157', 'OriginAirportID_10158', 'OriginAirportID_10165', 'OriginAirportID_10170', 'OriginAirportID_10185', 'OriginAirportID_10208', 'OriginAirportID_10245', 'OriginAirportID_10257', 'OriginAirportID_10268', 'OriginAirportID_10272', 'OriginAirportID_10275', 'OriginAirportID_10279', 'OriginAirportID_10299', 'OriginAirportID_10333', 'OriginAirportID_10361', 'OriginAirportID_10372', 'OriginAirportID_10397', 'OriginAirportID_10408', 'OriginAirportID_10423', 'OriginAirportID_10431', 'OriginAirportID_10434', 'OriginAirportID_10466', 'OriginAirportID_10469', 'OriginAirportID_10529', 'OriginAirportID_10551', 'OriginAirportID_10558', 'OriginAirportID_10561', 'OriginAirportID_10577', 'OriginAirportID_10581', 'OriginAirportID_10599', 'OriginAirportID_10617', 'OriginAirportID_10620', 'OriginAirportID_10627', 'OriginAirportID_10631', 'OriginAirportID_10643', 'OriginAirportID_10666', 'OriginAirportID_10676', 'OriginAirportID_10685', 'OriginAirportID_10693', 'OriginAirportID_10713', 'OriginAirportID_10721', 'OriginAirportID_10728', 'OriginAirportID_10731', 'OriginAirportID_10732', 'OriginAirportID_10739', 'OriginAirportID_10747', 'OriginAirportID_10754', 'OriginAirportID_10779', 'OriginAirportID_10781', 'OriginAirportID_10785', 'OriginAirportID_10792', 'OriginAirportID_10800', 'OriginAirportID_10821', 'OriginAirportID_10849', 'OriginAirportID_10868', 'OriginAirportID_10874', 'OriginAirportID_10918', 'OriginAirportID_10926', 'OriginAirportID_10967', 'OriginAirportID_10980', 'OriginAirportID_10990', 'OriginAirportID_10994', 'OriginAirportID_11003', 'OriginAirportID_11013', 'OriginAirportID_11027', 'OriginAirportID_11042', 'OriginAirportID_11049', 'OriginAirportID_11057', 'OriginAirportID_11066', 'OriginAirportID_11067', 'OriginAirportID_11076', 'OriginAirportID_11092', 'OriginAirportID_11097', 'OriginAirportID_11109', 'OriginAirportID_11111', 'OriginAirportID_11122', 'OriginAirportID_11140', 'OriginAirportID_11146', 'OriginAirportID_11150', 'OriginAirportID_11193', 'OriginAirportID_11203', 'OriginAirportID_11233', 'OriginAirportID_11252', 'OriginAirportID_11259', 'OriginAirportID_11267', 'OriginAirportID_11274', 'OriginAirportID_11278', 'OriginAirportID_11283', 'OriginAirportID_11288', 'OriginAirportID_11292', 'OriginAirportID_11298', 'OriginAirportID_11308', 'OriginAirportID_11315', 'OriginAirportID_11336', 'OriginAirportID_11337', 'OriginAirportID_11413', 'OriginAirportID_11415', 'OriginAirportID_11423', 'OriginAirportID_11433', 'OriginAirportID_11447', 'OriginAirportID_11468', 'OriginAirportID_11470', 'OriginAirportID_11471', 'OriginAirportID_11481', 'OriginAirportID_11503', 'OriginAirportID_11525', 'OriginAirportID_11537', 'OriginAirportID_11540', 'OriginAirportID_11577', 'OriginAirportID_11587', 'OriginAirportID_11603', 'OriginAirportID_11612', 'OriginAirportID_11617', 'OriginAirportID_11618', 'OriginAirportID_11624', 'OriginAirportID_11630', 'OriginAirportID_11637', 'OriginAirportID_11638', 'OriginAirportID_11641', 'OriginAirportID_11648', 'OriginAirportID_11695', 'OriginAirportID_11697', 'OriginAirportID_11699', 'OriginAirportID_11721', 'OriginAirportID_11725', 'OriginAirportID_11775', 'OriginAirportID_11778', 'OriginAirportID_11823', 'OriginAirportID_11865', 'OriginAirportID_11867', 'OriginAirportID_11884', 'OriginAirportID_11898', 'OriginAirportID_11905', 'OriginAirportID_11921', 'OriginAirportID_11953', 'OriginAirportID_11973', 'OriginAirportID_11977', 'OriginAirportID_11980', 'OriginAirportID_11982', 'OriginAirportID_11986', 'OriginAirportID_11995', 'OriginAirportID_11996', 'OriginAirportID_11997', 'OriginAirportID_12003', 'OriginAirportID_12007', 'OriginAirportID_12012', 'OriginAirportID_12016', 'OriginAirportID_12094', 'OriginAirportID_12119', 'OriginAirportID_12124', 'OriginAirportID_12129', 'OriginAirportID_12156', 'OriginAirportID_12173', 'OriginAirportID_12177', 'OriginAirportID_12191', 'OriginAirportID_12197', 'OriginAirportID_12206', 'OriginAirportID_12217', 'OriginAirportID_12223', 'OriginAirportID_12250', 'OriginAirportID_12255', 'OriginAirportID_12264', 'OriginAirportID_12265', 'OriginAirportID_12266', 'OriginAirportID_12278', 'OriginAirportID_12280', 'OriginAirportID_12320', 'OriginAirportID_12323', 'OriginAirportID_12335', 'OriginAirportID_12339', 'OriginAirportID_12343', 'OriginAirportID_12391', 'OriginAirportID_12397', 'OriginAirportID_12402', 'OriginAirportID_12441', 'OriginAirportID_12448', 'OriginAirportID_12451', 'OriginAirportID_12478', 'OriginAirportID_12511', 'OriginAirportID_12519', 'OriginAirportID_12523', 'OriginAirportID_12544', 'OriginAirportID_12559', 'OriginAirportID_12758', 'OriginAirportID_12819', 'OriginAirportID_12884', 'OriginAirportID_12888', 'OriginAirportID_12889', 'OriginAirportID_12891', 'OriginAirportID_12892', 'OriginAirportID_12896', 'OriginAirportID_12898', 'OriginAirportID_12899', 'OriginAirportID_12902', 'OriginAirportID_12915', 'OriginAirportID_12917', 'OriginAirportID_12945', 'OriginAirportID_12951', 'OriginAirportID_12953', 'OriginAirportID_12954', 'OriginAirportID_12982', 'OriginAirportID_12992', 'OriginAirportID_13029', 'OriginAirportID_13061', 'OriginAirportID_13076', 'OriginAirportID_13121', 'OriginAirportID_13127', 'OriginAirportID_13139', 'OriginAirportID_13158', 'OriginAirportID_13184', 'OriginAirportID_13198', 'OriginAirportID_13204', 'OriginAirportID_13211', 'OriginAirportID_13230', 'OriginAirportID_13232', 'OriginAirportID_13241', 'OriginAirportID_13244', 'OriginAirportID_13256', 'OriginAirportID_13264', 'OriginAirportID_13277', 'OriginAirportID_13290', 'OriginAirportID_13296', 'OriginAirportID_13303', 'OriginAirportID_13342', 'OriginAirportID_13344', 'OriginAirportID_13360', 'OriginAirportID_13367', 'OriginAirportID_13377', 'OriginAirportID_13422', 'OriginAirportID_13433', 'OriginAirportID_13459', 'OriginAirportID_13476', 'OriginAirportID_13485', 'OriginAirportID_13486', 'OriginAirportID_13487', 'OriginAirportID_13495', 'OriginAirportID_13502', 'OriginAirportID_13541', 'OriginAirportID_13577', 'OriginAirportID_13795', 'OriginAirportID_13796', 'OriginAirportID_13829', 'OriginAirportID_13830', 'OriginAirportID_13832', 'OriginAirportID_13851', 'OriginAirportID_13871', 'OriginAirportID_13873', 'OriginAirportID_13891', 'OriginAirportID_13930', 'OriginAirportID_13931', 'OriginAirportID_13933', 'OriginAirportID_13964', 'OriginAirportID_13970', 'OriginAirportID_13983', 'OriginAirportID_14004', 'OriginAirportID_14006', 'OriginAirportID_14025', 'OriginAirportID_14027', 'OriginAirportID_14057', 'OriginAirportID_14082', 'OriginAirportID_14092', 'OriginAirportID_14098', 'OriginAirportID_14100', 'OriginAirportID_14107', 'OriginAirportID_14108', 'OriginAirportID_14109', 'OriginAirportID_14112', 'OriginAirportID_14113', 'OriginAirportID_14120', 'OriginAirportID_14122', 'OriginAirportID_14150', 'OriginAirportID_14193', 'OriginAirportID_14222', 'OriginAirportID_14231', 'OriginAirportID_14237', 'OriginAirportID_14252', 'OriginAirportID_14254', 'OriginAirportID_14256', 'OriginAirportID_14259', 'OriginAirportID_14262', 'OriginAirportID_14288', 'OriginAirportID_14303', 'OriginAirportID_14307', 'OriginAirportID_14314', 'OriginAirportID_14321', 'OriginAirportID_14457', 'OriginAirportID_14487', 'OriginAirportID_14489', 'OriginAirportID_14492', 'OriginAirportID_14512', 'OriginAirportID_14520', 'OriginAirportID_14524', 'OriginAirportID_14534', 'OriginAirportID_14543', 'OriginAirportID_14570', 'OriginAirportID_14574', 'OriginAirportID_14576', 'OriginAirportID_14588', 'OriginAirportID_14633', 'OriginAirportID_14635', 'OriginAirportID_14674', 'OriginAirportID_14679', 'OriginAirportID_14683', 'OriginAirportID_14685', 'OriginAirportID_14689', 'OriginAirportID_14696', 'OriginAirportID_14698', 'OriginAirportID_14704', 'OriginAirportID_14709', 'OriginAirportID_14711', 'OriginAirportID_14716', 'OriginAirportID_14730', 'OriginAirportID_14747', 'OriginAirportID_14761', 'OriginAirportID_14771', 'OriginAirportID_14783', 'OriginAirportID_14794', 'OriginAirportID_14802', 'OriginAirportID_14812', 'OriginAirportID_14814', 'OriginAirportID_14828', 'OriginAirportID_14831', 'OriginAirportID_14842', 'OriginAirportID_14843', 'OriginAirportID_14869', 'OriginAirportID_14877', 'OriginAirportID_14893', 'OriginAirportID_14905', 'OriginAirportID_14908', 'OriginAirportID_14952', 'OriginAirportID_14955', 'OriginAirportID_14960', 'OriginAirportID_14986', 'OriginAirportID_15008', 'OriginAirportID_15016', 'OriginAirportID_15023', 'OriginAirportID_15024', 'OriginAirportID_15027', 'OriginAirportID_15041', 'OriginAirportID_15048', 'OriginAirportID_15070', 'OriginAirportID_15074', 'OriginAirportID_15096', 'OriginAirportID_15138', 'OriginAirportID_15249', 'OriginAirportID_15295', 'OriginAirportID_15304', 'OriginAirportID_15323', 'OriginAirportID_15356', 'OriginAirportID_15370', 'OriginAirportID_15376', 'OriginAirportID_15380', 'OriginAirportID_15389', 'OriginAirportID_15401', 'OriginAirportID_15411', 'OriginAirportID_15412', 'OriginAirportID_15569', 'OriginAirportID_15582', 'OriginAirportID_15607', 'OriginAirportID_15624', 'OriginAirportID_15841', 'OriginAirportID_15897', 'OriginAirportID_15919', 'OriginAirportID_15991', 'OriginAirportID_16101', 'OriginAirportID_16218', 'OriginAirportID_16869'
]

all_destairportids = [
    'DestAirportID_10135', 'DestAirportID_10136', 'DestAirportID_10140', 'DestAirportID_10141', 'DestAirportID_10146', 'DestAirportID_10154', 'DestAirportID_10155', 'DestAirportID_10157', 'DestAirportID_10158', 'DestAirportID_10165', 'DestAirportID_10170', 'DestAirportID_10185', 'DestAirportID_10208', 'DestAirportID_10245', 'DestAirportID_10257', 'DestAirportID_10268', 'DestAirportID_10272', 'DestAirportID_10275', 'DestAirportID_10279', 'DestAirportID_10299', 'DestAirportID_10333', 'DestAirportID_10361', 'DestAirportID_10372', 'DestAirportID_10397', 'DestAirportID_10408', 'DestAirportID_10423', 'DestAirportID_10431', 'DestAirportID_10434', 'DestAirportID_10466', 'DestAirportID_10469', 'DestAirportID_10529', 'DestAirportID_10551', 'DestAirportID_10558', 'DestAirportID_10561', 'DestAirportID_10577', 'DestAirportID_10581', 'DestAirportID_10599', 'DestAirportID_10617', 'DestAirportID_10620', 'DestAirportID_10627', 'DestAirportID_10631', 'DestAirportID_10643', 'DestAirportID_10666', 'DestAirportID_10676', 'DestAirportID_10685', 'DestAirportID_10693', 'DestAirportID_10713', 'DestAirportID_10721', 'DestAirportID_10728', 'DestAirportID_10731', 'DestAirportID_10732', 'DestAirportID_10739', 'DestAirportID_10747', 'DestAirportID_10754', 'DestAirportID_10779', 'DestAirportID_10781', 'DestAirportID_10785', 'DestAirportID_10792', 'DestAirportID_10800', 'DestAirportID_10821', 'DestAirportID_10849', 'DestAirportID_10868', 'DestAirportID_10874', 'DestAirportID_10918', 'DestAirportID_10926', 'DestAirportID_10967', 'DestAirportID_10980', 'DestAirportID_10990', 'DestAirportID_10994', 'DestAirportID_11003', 'DestAirportID_11013', 'DestAirportID_11027', 'DestAirportID_11042', 'DestAirportID_11049', 'DestAirportID_11057', 'DestAirportID_11066', 'DestAirportID_11067', 'DestAirportID_11076', 'DestAirportID_11092', 'DestAirportID_11097', 'DestAirportID_11109', 'DestAirportID_11111', 'DestAirportID_11122', 'DestAirportID_11140', 'DestAirportID_11146', 'DestAirportID_11150', 'DestAirportID_11193', 'DestAirportID_11203', 'DestAirportID_11233', 'DestAirportID_11252', 'DestAirportID_11259', 'DestAirportID_11267', 'DestAirportID_11274', 'DestAirportID_11278', 'DestAirportID_11283', 'DestAirportID_11288', 'DestAirportID_11292', 'DestAirportID_11298', 'DestAirportID_11308', 'DestAirportID_11315', 'DestAirportID_11336', 'DestAirportID_11337', 'DestAirportID_11413', 'DestAirportID_11415', 'DestAirportID_11423', 'DestAirportID_11433', 'DestAirportID_11447', 'DestAirportID_11468', 'DestAirportID_11470', 'DestAirportID_11471', 'DestAirportID_11481', 'DestAirportID_11503', 'DestAirportID_11525', 'DestAirportID_11537', 'DestAirportID_11540', 'DestAirportID_11577', 'DestAirportID_11587', 'DestAirportID_11603', 'DestAirportID_11612', 'DestAirportID_11617', 'DestAirportID_11618', 'DestAirportID_11624', 'DestAirportID_11630', 'DestAirportID_11637', 'DestAirportID_11638', 'DestAirportID_11641', 'DestAirportID_11648', 'DestAirportID_11695', 'DestAirportID_11697', 'DestAirportID_11699', 'DestAirportID_11721', 'DestAirportID_11725', 'DestAirportID_11775', 'DestAirportID_11778', 'DestAirportID_11823', 'DestAirportID_11865', 'DestAirportID_11867', 'DestAirportID_11884', 'DestAirportID_11898', 'DestAirportID_11905', 'DestAirportID_11921', 'DestAirportID_11953', 'DestAirportID_11973', 'DestAirportID_11977', 'DestAirportID_11980', 'DestAirportID_11982', 'DestAirportID_11986', 'DestAirportID_11995', 'DestAirportID_11996', 'DestAirportID_11997', 'DestAirportID_12003', 'DestAirportID_12007', 'DestAirportID_12012', 'DestAirportID_12016', 'DestAirportID_12094', 'DestAirportID_12119', 'DestAirportID_12124', 'DestAirportID_12129', 'DestAirportID_12156', 'DestAirportID_12173', 'DestAirportID_12177', 'DestAirportID_12191', 'DestAirportID_12197', 'DestAirportID_12206', 'DestAirportID_12217', 'DestAirportID_12223', 'DestAirportID_12250', 'DestAirportID_12255', 'DestAirportID_12264', 'DestAirportID_12265', 'DestAirportID_12266', 'DestAirportID_12278', 'DestAirportID_12280', 'DestAirportID_12320', 'DestAirportID_12323', 'DestAirportID_12335', 'DestAirportID_12339', 'DestAirportID_12343', 'DestAirportID_12391', 'DestAirportID_12397', 'DestAirportID_12402', 'DestAirportID_12441', 'DestAirportID_12448', 'DestAirportID_12451', 'DestAirportID_12478', 'DestAirportID_12511', 'DestAirportID_12519', 'DestAirportID_12523', 'DestAirportID_12544', 'DestAirportID_12559', 'DestAirportID_12758', 'DestAirportID_12819', 'DestAirportID_12884', 'DestAirportID_12888', 'DestAirportID_12889', 'DestAirportID_12891', 'DestAirportID_12892', 'DestAirportID_12896', 'DestAirportID_12898', 'DestAirportID_12899', 'DestAirportID_12902', 'DestAirportID_12915', 'DestAirportID_12917', 'DestAirportID_12945', 'DestAirportID_12951', 'DestAirportID_12953', 'DestAirportID_12954', 'DestAirportID_12982', 'DestAirportID_12992', 'DestAirportID_13029', 'DestAirportID_13061', 'DestAirportID_13076', 'DestAirportID_13121', 'DestAirportID_13127', 'DestAirportID_13139', 'DestAirportID_13158', 'DestAirportID_13184', 'DestAirportID_13198', 'DestAirportID_13204', 'DestAirportID_13211', 'DestAirportID_13230', 'DestAirportID_13232', 'DestAirportID_13241', 'DestAirportID_13244', 'DestAirportID_13256', 'DestAirportID_13264', 'DestAirportID_13277', 'DestAirportID_13290', 'DestAirportID_13296', 'DestAirportID_13303', 'DestAirportID_13342', 'DestAirportID_13344', 'DestAirportID_13360', 'DestAirportID_13367', 'DestAirportID_13377', 'DestAirportID_13422', 'DestAirportID_13433', 'DestAirportID_13459', 'DestAirportID_13476', 'DestAirportID_13485', 'DestAirportID_13486', 'DestAirportID_13487', 'DestAirportID_13495', 'DestAirportID_13502', 'DestAirportID_13541', 'DestAirportID_13577', 'DestAirportID_13795', 'DestAirportID_13796', 'DestAirportID_13829', 'DestAirportID_13830', 'DestAirportID_13832', 'DestAirportID_13851', 'DestAirportID_13871', 'DestAirportID_13873', 'DestAirportID_13891', 'DestAirportID_13930', 'DestAirportID_13931', 'DestAirportID_13933', 'DestAirportID_13964', 'DestAirportID_13970', 'DestAirportID_13983', 'DestAirportID_14004', 'DestAirportID_14006', 'DestAirportID_14025', 'DestAirportID_14027', 'DestAirportID_14057', 'DestAirportID_14082', 'DestAirportID_14092', 'DestAirportID_14098', 'DestAirportID_14100', 'DestAirportID_14107', 'DestAirportID_14108', 'DestAirportID_14109', 'DestAirportID_14112', 'DestAirportID_14113', 'DestAirportID_14120', 'DestAirportID_14122', 'DestAirportID_14150', 'DestAirportID_14193', 'DestAirportID_14222', 'DestAirportID_14231', 'DestAirportID_14237', 'DestAirportID_14252', 'DestAirportID_14254', 'DestAirportID_14256', 'DestAirportID_14259', 'DestAirportID_14262', 'DestAirportID_14288', 'DestAirportID_14303', 'DestAirportID_14307', 'DestAirportID_14314', 'DestAirportID_14321', 'DestAirportID_14457', 'DestAirportID_14487', 'DestAirportID_14489', 'DestAirportID_14492', 'DestAirportID_14512', 'DestAirportID_14520', 'DestAirportID_14524', 'DestAirportID_14534', 'DestAirportID_14543', 'DestAirportID_14570', 'DestAirportID_14574', 'DestAirportID_14576', 'DestAirportID_14588', 'DestAirportID_14633', 'DestAirportID_14635', 'DestAirportID_14674', 'DestAirportID_14679', 'DestAirportID_14683', 'DestAirportID_14685', 'DestAirportID_14689', 'DestAirportID_14696', 'DestAirportID_14698', 'DestAirportID_14704', 'DestAirportID_14709', 'DestAirportID_14711', 'DestAirportID_14716', 'DestAirportID_14730', 'DestAirportID_14747', 'DestAirportID_14761', 'DestAirportID_14771', 'DestAirportID_14783', 'DestAirportID_14794', 'DestAirportID_14802', 'DestAirportID_14812', 'DestAirportID_14814', 'DestAirportID_14828', 'DestAirportID_14831', 'DestAirportID_14842', 'DestAirportID_14843', 'DestAirportID_14869', 'DestAirportID_14877', 'DestAirportID_14893', 'DestAirportID_14905', 'DestAirportID_14908', 'DestAirportID_14952', 'DestAirportID_14955', 'DestAirportID_14960', 'DestAirportID_14986', 'DestAirportID_15008', 'DestAirportID_15016', 'DestAirportID_15023', 'DestAirportID_15024', 'DestAirportID_15027', 'DestAirportID_15041', 'DestAirportID_15048', 'DestAirportID_15070', 'DestAirportID_15074', 'DestAirportID_15096', 'DestAirportID_15138', 'DestAirportID_15249', 'DestAirportID_15295', 'DestAirportID_15304', 'DestAirportID_15323', 'DestAirportID_15356', 'DestAirportID_15370', 'DestAirportID_15376', 'DestAirportID_15380', 'DestAirportID_15389', 'DestAirportID_15401', 'DestAirportID_15411', 'DestAirportID_15412', 'DestAirportID_15569', 'DestAirportID_15582', 'DestAirportID_15607', 'DestAirportID_15624', 'DestAirportID_15841', 'DestAirportID_15897', 'DestAirportID_15919', 'DestAirportID_15991', 'DestAirportID_16101', 'DestAirportID_16218', 'DestAirportID_16869'
]


rf_model = joblib.load('rf_model_gini.joblib')

airport_codes = [
    'Origin__ABE', 'Origin__ABI', 'Origin__ABQ', 'Origin__ABR', 'Origin__ABY',
    'Origin__ACK', 'Origin__ACT', 'Origin__ACV', 'Origin__ACY', 'Origin__ADK',
    'Origin__ADQ', 'Origin__AEX', 'Origin__AGS', 'Origin__AKN', 'Origin__ALB',
    'Origin__ALO', 'Origin__ALS', 'Origin__ALW', 'Origin__AMA', 'Origin__ANC',
    'Origin__APN', 'Origin__ART', 'Origin__ASE', 'Origin__ATL', 'Origin__ATW',
    'Origin__ATY', 'Origin__AUS', 'Origin__AVL', 'Origin__AVP', 'Origin__AZA',
    'Origin__AZO', 'Origin__BDL', 'Origin__BET', 'Origin__BFF', 'Origin__BFL',
    'Origin__BGM', 'Origin__BGR', 'Origin__BIH', 'Origin__BIL', 'Origin__BIS',
    'Origin__BJI', 'Origin__BKG', 'Origin__BLI', 'Origin__BLV', 'Origin__BMI',
    'Origin__BNA', 'Origin__BOI', 'Origin__BOS', 'Origin__BPT', 'Origin__BQK',
    'Origin__BQN', 'Origin__BRD', 'Origin__BRO', 'Origin__BRW', 'Origin__BTM',
    'Origin__BTR', 'Origin__BTV', 'Origin__BUF', 'Origin__BUR', 'Origin__BWI',
    'Origin__BZN', 'Origin__CAE', 'Origin__CAK', 'Origin__CDB', 'Origin__CDC',
    'Origin__CDV', 'Origin__CGI', 'Origin__CHA', 'Origin__CHO', 'Origin__CHS',
    'Origin__CID', 'Origin__CIU', 'Origin__CKB', 'Origin__CLE', 'Origin__CLL',
    'Origin__CLT', 'Origin__CMH', 'Origin__CMI', 'Origin__CMX', 'Origin__CNY',
    'Origin__COD', 'Origin__COS', 'Origin__COU', 'Origin__CPR', 'Origin__CRP',
    'Origin__CRW', 'Origin__CSG', 'Origin__CVG', 'Origin__CWA', 'Origin__CYS',
    'Origin__DAB', 'Origin__DAL', 'Origin__DAY', 'Origin__DBQ', 'Origin__DCA',
    'Origin__DDC', 'Origin__DEC', 'Origin__DEN', 'Origin__DFW', 'Origin__DIK',
    'Origin__DLG', 'Origin__DLH', 'Origin__DRO', 'Origin__DRT', 'Origin__DSM',
    'Origin__DTW', 'Origin__DUT', 'Origin__DVL', 'Origin__EAR', 'Origin__EAT',
    'Origin__EAU', 'Origin__ECP', 'Origin__EGE', 'Origin__EKO', 'Origin__ELM',
    'Origin__ELP', 'Origin__ERI', 'Origin__ESC', 'Origin__EUG', 'Origin__EVV',
    'Origin__EWN', 'Origin__EWR', 'Origin__EYW', 'Origin__FAI', 'Origin__FAR',
    'Origin__FAT', 'Origin__FAY', 'Origin__FCA', 'Origin__FLG', 'Origin__FLL',
    'Origin__FLO', 'Origin__FNT', 'Origin__FOD', 'Origin__FSD', 'Origin__FSM',
    'Origin__FWA', 'Origin__GCC', 'Origin__GCK', 'Origin__GEG', 'Origin__GFK',
    'Origin__GGG', 'Origin__GJT', 'Origin__GNV', 'Origin__GPT', 'Origin__GRB',
    'Origin__GRI', 'Origin__GRK', 'Origin__GRR', 'Origin__GSO', 'Origin__GSP',
    'Origin__GST', 'Origin__GTF', 'Origin__GTR', 'Origin__GUC', 'Origin__GUM',
    'Origin__HDN', 'Origin__HGR', 'Origin__HHH', 'Origin__HIB', 'Origin__HLN',
    'Origin__HOB', 'Origin__HOU', 'Origin__HPN', 'Origin__HRL', 'Origin__HTS',
    'Origin__HVN', 'Origin__HYA', 'Origin__HYS', 'Origin__IAD', 'Origin__IAG',
    'Origin__IAH', 'Origin__ICT', 'Origin__IDA', 'Origin__ILG', 'Origin__ILM',
    'Origin__IMT', 'Origin__IND', 'Origin__INL', 'Origin__IPT', 'Origin__ISN',
    'Origin__ISP', 'Origin__ITH', 'Origin__JAC', 'Origin__JAN', 'Origin__JAX',
    'Origin__JFK', 'Origin__JLN', 'Origin__JMS', 'Origin__JNU', 'Origin__JST',
    'Origin__KTN', 'Origin__LAN', 'Origin__LAR', 'Origin__LAS', 'Origin__LAW',
    'Origin__LAX', 'Origin__LBB', 'Origin__LBE', 'Origin__LBF', 'Origin__LBL',
    'Origin__LCH', 'Origin__LCK', 'Origin__LEX', 'Origin__LFT', 'Origin__LGA',
    'Origin__LGB', 'Origin__LIT', 'Origin__LNK', 'Origin__LRD', 'Origin__LSE',
    'Origin__LWB', 'Origin__LWS', 'Origin__LYH', 'Origin__MAF', 'Origin__MBS',
    'Origin__MCI', 'Origin__MCO', 'Origin__MCW', 'Origin__MDT', 'Origin__MDW',
    'Origin__MEI', 'Origin__MEM', 'Origin__MFE', 'Origin__MFR', 'Origin__MHK',
    'Origin__MHT', 'Origin__MIA', 'Origin__MKE', 'Origin__MKG', 'Origin__MLB',
    'Origin__MLI', 'Origin__MLU', 'Origin__MMH', 'Origin__MOT', 'Origin__MQT',
    'Origin__MRY', 'Origin__MSN', 'Origin__MSO', 'Origin__MSP', 'Origin__MSY',
    'Origin__MTJ', 'Origin__MVY', 'Origin__MYR', 'Origin__OAJ', 'Origin__OAK',
    'Origin__OGD', 'Origin__OGS', 'Origin__OKC', 'Origin__OMA', 'Origin__OME',
    'Origin__ONT', 'Origin__ORD', 'Origin__ORF', 'Origin__ORH', 'Origin__OTH',
    'Origin__OTZ', 'Origin__OWB', 'Origin__PAE', 'Origin__PAH', 'Origin__PBG',
    'Origin__PBI', 'Origin__PDX', 'Origin__PGD', 'Origin__PGV', 'Origin__PHF',
    'Origin__PHL', 'Origin__PHX', 'Origin__PIA', 'Origin__PIB', 'Origin__PIE',
    'Origin__PIH', 'Origin__PIR', 'Origin__PIT', 'Origin__PLN', 'Origin__PNS',
    'Origin__PQI', 'Origin__PRC', 'Origin__PSC', 'Origin__PSE', 'Origin__PSG',
    'Origin__PSM', 'Origin__PSP', 'Origin__PUB', 'Origin__PUW', 'Origin__PVD',
    'Origin__PVU', 'Origin__PWM', 'Origin__RAP', 'Origin__RDD', 'Origin__RDM',
    'Origin__RDU', 'Origin__RFD', 'Origin__RHI', 'Origin__RIC', 'Origin__RIW',
    'Origin__RKS', 'Origin__RNO', 'Origin__ROA', 'Origin__ROC', 'Origin__ROP',
    'Origin__ROW', 'Origin__RST', 'Origin__RSW', 'Origin__SAF', 'Origin__SAN',
    'Origin__SAT', 'Origin__SAV', 'Origin__SBA', 'Origin__SBN', 'Origin__SBP',
    'Origin__SBY', 'Origin__SCC', 'Origin__SCE', 'Origin__SCK', 'Origin__SDF',
    'Origin__SEA', 'Origin__SFB', 'Origin__SFO', 'Origin__SGF', 'Origin__SGU',
    'Origin__SHD', 'Origin__SHR', 'Origin__SHV', 'Origin__SIT', 'Origin__SJC',
    'Origin__SJT', 'Origin__SJU', 'Origin__SLC', 'Origin__SLN', 'Origin__SMF',
    'Origin__SMX', 'Origin__SNA', 'Origin__SPI', 'Origin__SPN', 'Origin__SPS',
    'Origin__SRQ', 'Origin__STC', 'Origin__STL', 'Origin__STS', 'Origin__STT',
    'Origin__STX', 'Origin__SUN', 'Origin__SUX', 'Origin__SWF', 'Origin__SWO',
    'Origin__SYR', 'Origin__TBN', 'Origin__TLH', 'Origin__TOL', 'Origin__TPA',
    'Origin__TRI', 'Origin__TTN', 'Origin__TUL', 'Origin__TUS', 'Origin__TVC',
    'Origin__TWF', 'Origin__TXK', 'Origin__TYR', 'Origin__TYS', 'Origin__UIN',
    'Origin__USA', 'Origin__VCT', 'Origin__VEL', 'Origin__VLD', 'Origin__VPS',
    'Origin__WRG', 'Origin__WYS', 'Origin__XNA', 'Origin__XWA', 'Origin__YAK',
    'Origin__YKM', 'Origin__YNG', 'Origin__YUM'
]

airline_names = [
    'Airline__Air Wisconsin Airlines Corp',
    'Airline__Alaska Airlines Inc.',
    'Airline__Allegiant Air',
    'Airline__American Airlines Inc.',
    'Airline__Cape Air',
    'Airline__Capital Cargo International',
    'Airline__Comair Inc.',
    'Airline__Commutair Aka Champlain Enterprises, Inc.',
    'Airline__Compass Airlines',
    'Airline__Delta Air Lines Inc.',
    'Airline__Endeavor Air Inc.',
    'Airline__Envoy Air',
    'Airline__ExpressJet Airlines Inc.',
    'Airline__Frontier Airlines Inc.',
    'Airline__GoJet Airlines, LLC d/b/a United Express',
    'Airline__Horizon Air',
    'Airline__JetBlue Airways',
    'Airline__Mesa Airlines Inc.',
    'Airline__Peninsula Airways Inc.',
    'Airline__Republic Airlines',
    'Airline__SkyWest Airlines Inc.',
    'Airline__Southwest Airlines Co.',
    'Airline__Spirit Air Lines',
    'Airline__Trans States Airlines',
    'Airline__United Air Lines Inc.',
    'Airline__Virgin America'
]

destination_codes = [
    'Dest__ABE', 'Dest__ABI', 'Dest__ABQ', 'Dest__ABR', 'Dest__ABY',
    'Dest__ACK', 'Dest__ACT', 'Dest__ACV', 'Dest__ACY', 'Dest__ADK',
    'Dest__ADQ', 'Dest__AEX', 'Dest__AGS', 'Dest__AKN', 'Dest__ALB',
    'Dest__ALO', 'Dest__ALS', 'Dest__ALW', 'Dest__AMA', 'Dest__ANC',
    'Dest__APN', 'Dest__ART', 'Dest__ASE', 'Dest__ATL', 'Dest__ATW',
    'Dest__ATY', 'Dest__AUS', 'Dest__AVL', 'Dest__AVP', 'Dest__AZA',
    'Dest__AZO', 'Dest__BDL', 'Dest__BET', 'Dest__BFF', 'Dest__BFL',
    'Dest__BGM', 'Dest__BGR', 'Dest__BIH', 'Dest__BIL', 'Dest__BIS',
    'Dest__BJI', 'Dest__BKG', 'Dest__BLI', 'Dest__BLV', 'Dest__BMI',
    'Dest__BNA', 'Dest__BOI', 'Dest__BOS', 'Dest__BPT', 'Dest__BQK',
    'Dest__BQN', 'Dest__BRD', 'Dest__BRO', 'Dest__BRW', 'Dest__BTM',
    'Dest__BTR', 'Dest__BTV', 'Dest__BUF', 'Dest__BUR', 'Dest__BWI',
    'Dest__BZN', 'Dest__CAE', 'Dest__CAK', 'Dest__CDB', 'Dest__CDC',
    'Dest__CDV', 'Dest__CGI', 'Dest__CHA', 'Dest__CHO', 'Dest__CHS',
    'Dest__CID', 'Dest__CIU', 'Dest__CKB', 'Dest__CLE', 'Dest__CLL',
    'Dest__CLT', 'Dest__CMH', 'Dest__CMI', 'Dest__CMX', 'Dest__CNY',
    'Dest__COD', 'Dest__COS', 'Dest__COU', 'Dest__CPR', 'Dest__CRP',
    'Dest__CRW', 'Dest__CSG', 'Dest__CVG', 'Dest__CWA', 'Dest__CYS',
    'Dest__DAB', 'Dest__DAL', 'Dest__DAY', 'Dest__DBQ', 'Dest__DCA',
    'Dest__DDC', 'Dest__DEC', 'Dest__DEN', 'Dest__DFW', 'Dest__DIK',
    'Dest__DLG', 'Dest__DLH', 'Dest__DRO', 'Dest__DRT', 'Dest__DSM',
    'Dest__DTW', 'Dest__DUT', 'Dest__DVL', 'Dest__EAR', 'Dest__EAT',
    'Dest__EAU', 'Dest__ECP', 'Dest__EGE', 'Dest__EKO', 'Dest__ELM',
    'Dest__ELP', 'Dest__ERI', 'Dest__ESC', 'Dest__EUG', 'Dest__EVV',
    'Dest__EWN', 'Dest__EWR', 'Dest__EYW', 'Dest__FAI', 'Dest__FAR',
    'Dest__FAT', 'Dest__FAY', 'Dest__FCA', 'Dest__FLG', 'Dest__FLL',
    'Dest__FLO', 'Dest__FNT', 'Dest__FOD', 'Dest__FSD', 'Dest__FSM',
    'Dest__FWA', 'Dest__GCC', 'Dest__GCK', 'Dest__GEG', 'Dest__GFK',
    'Dest__GGG', 'Dest__GJT', 'Dest__GNV', 'Dest__GPT', 'Dest__GRB',
    'Dest__GRI', 'Dest__GRK', 'Dest__GRR', 'Dest__GSO', 'Dest__GSP',
    'Dest__GST', 'Dest__GTF', 'Dest__GTR', 'Dest__GUC', 'Dest__GUM',
    'Dest__HDN', 'Dest__HGR', 'Dest__HHH', 'Dest__HIB', 'Dest__HLN',
    'Dest__HOB', 'Dest__HOU', 'Dest__HPN', 'Dest__HRL', 'Dest__HTS',
    'Dest__HVN', 'Dest__HYA', 'Dest__HYS', 'Dest__IAD', 'Dest__IAG',
    'Dest__IAH', 'Dest__ICT', 'Dest__IDA', 'Dest__ILG', 'Dest__ILM',
    'Dest__IMT', 'Dest__IND', 'Dest__INL', 'Dest__IPT', 'Dest__ISN',
    'Dest__ISP', 'Dest__ITH', 'Dest__JAC', 'Dest__JAN', 'Dest__JAX',
    'Dest__JFK', 'Dest__JLN', 'Dest__JMS', 'Dest__JNU', 'Dest__JST',
    'Dest__KTN', 'Dest__LAN', 'Dest__LAR', 'Dest__LAS', 'Dest__LAW',
    'Dest__LAX', 'Dest__LBB', 'Dest__LBE', 'Dest__LBF', 'Dest__LBL',
    'Dest__LCH', 'Dest__LCK', 'Dest__LEX', 'Dest__LFT', 'Dest__LGA',
    'Dest__LGB', 'Dest__LIT', 'Dest__LNK', 'Dest__LRD', 'Dest__LSE',
    'Dest__LWB', 'Dest__LWS', 'Dest__LYH', 'Dest__MAF', 'Dest__MBS',
    'Dest__MCI', 'Dest__MCO', 'Dest__MCW', 'Dest__MDT', 'Dest__MDW',
    'Dest__MEI', 'Dest__MEM', 'Dest__MFE', 'Dest__MFR', 'Dest__MHK',
    'Dest__MHT', 'Dest__MIA', 'Dest__MKE', 'Dest__MKG', 'Dest__MLB',
    'Dest__MLI', 'Dest__MLU', 'Dest__MMH', 'Dest__MOT', 'Dest__MQT',
    'Dest__MRY', 'Dest__MSN', 'Dest__MSO', 'Dest__MSP', 'Dest__MSY',
    'Dest__MTJ', 'Dest__MVY', 'Dest__MYR', 'Dest__OAJ', 'Dest__OAK',
    'Dest__OGD', 'Dest__OGS', 'Dest__OKC', 'Dest__OMA', 'Dest__OME',
    'Dest__ONT', 'Dest__ORD', 'Dest__ORF', 'Dest__ORH', 'Dest__OTH',
    'Dest__OTZ', 'Dest__OWB', 'Dest__PAE', 'Dest__PAH', 'Dest__PBG',
    'Dest__PBI', 'Dest__PDX', 'Dest__PGD', 'Dest__PGV', 'Dest__PHF',
    'Dest__PHL', 'Dest__PHX', 'Dest__PIA', 'Dest__PIB', 'Dest__PIE',
    'Dest__PIH', 'Dest__PIR', 'Dest__PIT', 'Dest__PLN', 'Dest__PNS',
    'Dest__PQI', 'Dest__PRC', 'Dest__PSC', 'Dest__PSE', 'Dest__PSG',
    'Dest__PSM', 'Dest__PSP', 'Dest__PUB', 'Dest__PUW', 'Dest__PVD',
    'Dest__PVU', 'Dest__PWM', 'Dest__RAP', 'Dest__RDD', 'Dest__RDM',
    'Dest__RDU', 'Dest__RFD', 'Dest__RHI', 'Dest__RIC', 'Dest__RIW',
    'Dest__RKS', 'Dest__RNO', 'Dest__ROA', 'Dest__ROC', 'Dest__ROP',
    'Dest__ROW', 'Dest__RST', 'Dest__RSW', 'Dest__SAF', 'Dest__SAN',
    'Dest__SAT', 'Dest__SAV', 'Dest__SBA', 'Dest__SBN', 'Dest__SBP',
    'Dest__SBY', 'Dest__SCC', 'Dest__SCE', 'Dest__SCK', 'Dest__SDF',
    'Dest__SEA', 'Dest__SFB', 'Dest__SFO', 'Dest__SGF', 'Dest__SGU',
    'Dest__SHD', 'Dest__SHR', 'Dest__SHV', 'Dest__SIT', 'Dest__SJC',
    'Dest__SJT', 'Dest__SJU', 'Dest__SLC', 'Dest__SLN', 'Dest__SMF',
    'Dest__SMX', 'Dest__SNA', 'Dest__SPI', 'Dest__SPN', 'Dest__SPS',
    'Dest__SRQ', 'Dest__STC', 'Dest__STL', 'Dest__STS', 'Dest__STT',
    'Dest__STX', 'Dest__SUN', 'Dest__SUX', 'Dest__SWF', 'Dest__SWO',
    'Dest__SYR', 'Dest__TBN', 'Dest__TLH', 'Dest__TOL', 'Dest__TPA',
    'Dest__TRI', 'Dest__TTN', 'Dest__TUL', 'Dest__TUS', 'Dest__TVC',
    'Dest__TWF', 'Dest__TXK', 'Dest__TYR', 'Dest__TYS', 'Dest__UIN',
    'Dest__USA', 'Dest__VCT', 'Dest__VEL', 'Dest__VLD', 'Dest__VPS',
    'Dest__WRG', 'Dest__WYS', 'Dest__XNA', 'Dest__XWA', 'Dest__YAK',
    'Dest__YKM', 'Dest__YNG', 'Dest__YUM'
]

@app.route('/')
def index():
    return 'Hello, World!'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Handle one-hot encoded features
    airlines = data.get('Airline', [])  # Array of selected airlines
    airline_features = [1 if airline in airlines else 0 for airline in all_airlines]

    months = data.get('Month', [])  # Array of selected months
    month_features = [1 if month in months else 0 for month in all_months]

    daysofmonth = data.get('DayofMonth', [])  # Array of selected daysofmonth
    dayofmonth_features = [1 if dayofmonth in daysofmonth else 0 for dayofmonth in all_daysofmonth]

    daysofweek = data.get('DayOfWeek', [])  # Array of selected daysofmonth
    dayofweek_features = [1 if dayofweek in daysofweek else 0 for dayofweek in all_daysofweek]

    originairportids = data.get('OriginAirportID', [])  # Array of selected daysofmonth
    originairportid_features = [1 if originairportid in originairportids else 0 for originairportid in all_originairportids]

    destairportids = data.get('DestAirportID', [])  # Array of selected daysofmonth
    destairportid_features = [1 if destairportid in destairportids else 0 for destairportid in all_destairportids]
    
    # Extract other non-one-hot-encoded features
    dep_time = float(data['DepTime'])
    dep_delay = float(data['DepDelay'])
    distance = float(data['Distance'])
    taxi_out = float(data['TaxiOut'])
    wheels_off = float(data['WheelsOff'])
    # arr_del15 = float(data['ArrDel15'])
    
    # Combine all features
    features = np.array([dep_time, dep_delay, distance, taxi_out, wheels_off] + airline_features + month_features + dayofmonth_features + dayofweek_features + originairportid_features + destairportid_features)

    
    # Convert the NumPy array to a PyTorch tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Make a prediction using your model
    prediction = model.predict(features_tensor)

    if prediction.item() == 0:
        return {'prediction': "Flight will most likely not be delayed"}
    else:
        return {'prediction': "Flight will most likely be delayed"}

    return {'prediction': prediction.item()}
    
    return jsonify({'prediction': prediction[0]})


@app.route('/predict_RF', methods=['POST'])
def predict_rf():
    data = request.json

if __name__ == '__main__':
    app.run(debug=True)
