"""
Modern theme and styling for the fraud analytics dashboard.
Implements contemporary design patterns and color schemes.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ModernTheme:
    """Modern design theme configuration."""
    
    # Modern color palette
    COLORS = {
        'primary': '#2E86C1',      # Professional blue
        'secondary': '#F39C12',     # Warm orange
        'success': '#27AE60',       # Fresh green
        'danger': '#E74C3C',        # Alert red
        'warning': '#F1C40F',       # Bright yellow
        'info': '#8E44AD',          # Purple
        'dark': '#2C3E50',          # Dark blue-gray
        'light': '#ECF0F1',         # Light gray
        'gradient_start': '#667eea', # Gradient blue
        'gradient_end': '#764ba2',   # Gradient purple
        'fraud': '#FF6B6B',         # Fraud red
        'legitimate': '#4ECDC4',    # Legitimate teal
        'risk_low': '#2ECC71',      # Low risk green
        'risk_medium': '#F39C12',   # Medium risk orange
        'risk_high': '#E74C3C',     # High risk red
        'background': '#FAFBFC',    # Subtle background
        'card_bg': '#FFFFFF',       # Card background
        'text_primary': '#2C3E50',  # Primary text
        'text_secondary': '#7F8C8D', # Secondary text
    }
    
    # Chart color sequences
    CHART_COLORS = [
        '#2E86C1', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD',
        '#16A085', '#D35400', '#7D3C98', '#148F77', '#B7950B'
    ]
    
    RISK_COLORS = {
        'low': '#2ECC71',
        'medium': '#F39C12', 
        'high': '#E74C3C'
    }
    
    @staticmethod
    def apply_custom_css():
        """Apply simplified modern CSS styling for faster loading."""
        st.markdown("""
        <style>
        /* Essential styling only for fast loading */
        [data-testid="metric-container"] {
            background: white;
            border: none;
            padding: 1rem;
            border-radius: 8px;
            border-left: 3px solid #2E86C1;
        }
        
        [data-testid="metric-container"] [data-testid="metric-value"] {
            font-size: 1.8rem;
            font-weight: 600;
            color: #2E86C1;
        }
        
        .stButton > button {
            background: #2E86C1;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.4rem 1rem;
        }
        
        .stButton > button:hover {
            background: #1F5F99;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def get_plotly_theme():
        """Get modern Plotly theme configuration."""
        return {
            'layout': {
                'colorway': ModernTheme.CHART_COLORS,
                'font': {
                    'family': 'Inter, sans-serif',
                    'size': 12,
                    'color': ModernTheme.COLORS['text_primary']
                },
                'title': {
                    'font': {
                        'family': 'Inter, sans-serif',
                        'size': 18,
                        'color': ModernTheme.COLORS['text_primary']
                    },
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'paper_bgcolor': 'white',
                'plot_bgcolor': 'white',
                'margin': {'l': 60, 'r': 60, 't': 80, 'b': 60},
                'hovermode': 'closest',
                'hoverlabel': {
                    'bgcolor': 'white',
                    'bordercolor': ModernTheme.COLORS['primary'],
                    'font': {'color': ModernTheme.COLORS['text_primary']}
                },
                'xaxis': {
                    'gridcolor': '#F0F0F0',
                    'linecolor': '#E0E0E0',
                    'tickcolor': '#E0E0E0',
                    'title': {'font': {'size': 14}}
                },
                'yaxis': {
                    'gridcolor': '#F0F0F0',
                    'linecolor': '#E0E0E0',
                    'tickcolor': '#E0E0E0',
                    'title': {'font': {'size': 14}}
                }
            }
        }
    
    @staticmethod
    def create_modern_card(title: str, content: str, color: str = 'primary'):
        """Create a modern card component."""
        card_color = ModernTheme.COLORS.get(color, ModernTheme.COLORS['primary'])
        
        st.markdown(f"""
        <div style="
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            border-left: 4px solid {card_color};
            margin: 1rem 0;
        ">
            <h3 style="color: {card_color}; margin-bottom: 0.5rem; font-size: 1.2rem;">{title}</h3>
            <p style="color: #2C3E50; margin: 0; line-height: 1.5;">{content}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_gradient_header(title: str, subtitle: str = ""):
        """Create a gradient header."""
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        ">
            <h1 style="
                color: white;
                margin: 0;
                font-size: 2.5rem;
                font-weight: 700;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">{title}</h1>
            {f'<p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)


def apply_modern_plotly_styling(fig):
    """Apply modern styling to Plotly figures."""
    theme = ModernTheme.get_plotly_theme()
    
    fig.update_layout(
        **theme['layout'],
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        )
    )
    
    # Add subtle grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    return fig
