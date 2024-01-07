param(
    [string]$mod
)

.\myenv\Scripts\activate
python cog_openai_api.py --model $mod
