import os
import sys
import datetime
import warnings
import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register Arial font with full Turkish character support
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
if os.path.exists(r"C:\Windows\Fonts\arial.ttf"):
    try:
        pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
        pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
        pdfmetrics.registerFont(TTFont("Arial-Italic", r"C:\Windows\Fonts\ariali.ttf"))
        FONT_REGULAR = "Arial"
        FONT_BOLD = "Arial-Bold"
        print("Arial font successfully registered for native Turkish typography!")
    except Exception as e:
        print("Font registration notice:", e)

# Page Numbering Canvas
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def draw_page_elements(self, page_count):
        self.saveState()
        self.setFont(FONT_REGULAR, 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(40, 815, "DIN 1.2738 Frezeleme | Otomatik ML ve Çok Amaçlı Optimizasyon Raporu")
            self.drawRightString(555, 815, "TÜBİTAK 2209/B Doğrulama")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(40, 810, 555, 810)

        # Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(40, 35, 555, 35)
        
        tarih_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
        self.drawString(40, 24, f"Oluşturulma: {tarih_str} | Otomatik ML & Karar Destek Sistemi v2.5")
        page_str = f"Sayfa {self._pageNumber} / {page_count}"
        self.drawRightString(555, 24, page_str)
        self.restoreState()

print("Setup completed successfully.")
