# GitHub Actions Setup for Daily Backtest

This document explains how to configure GitHub Actions to automatically run daily market backtests and commit results to your repository.

## Prerequisites

1. Your repository must have the following files:
   - `backtest.py` - The backtest script
   - `pyproject.toml` and `poetry.lock` - Poetry dependency files
   - `.github/workflows/daily-backtest.yml` - The workflow file

## Setup Instructions

### 1. Configure Repository Secrets

Go to your GitHub repository and add the following secrets:

**Path**: `Settings` → `Secrets and variables` → `Actions` → `New repository secret`

Add these four secrets:

| Secret Name | Description |
|-------------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS Access Key for Bedrock API |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Key for Bedrock API |
| `OPENAI_API_KEY` | OpenAI API Key for GPT models |
| `GEMINI_API_KEY` | Google Gemini API Key |

### 2. Configure Repository Permissions

To allow the action to commit changes:

1. Go to `Settings` → `Actions` → `General`
2. Under **Workflow permissions**, select:
   - **Read and write permissions**
   - **Allow GitHub Actions to create and approve pull requests** (optional)

### 3. Schedule Configuration

The workflow is configured to run:

- **When**: Monday to Friday at 9:30 AM UTC
- **Why**: After major market opening times
- **Frequency**: Daily (business days only)

To modify the schedule, edit the cron expression in `.github/workflows/daily-backtest.yml`:

```yaml
schedule:
  - cron: '30 9 * * 1-5'  # MM HH DD MM DOW
```

**Cron format**:

- `30` - Minute (30)
- `9` - Hour (9 AM UTC)
- `*` - Any day of month
- `*` - Any month
- `1-5` - Monday to Friday

### 4. Manual Trigger

The workflow can also be triggered manually:

1. Go to `Actions` tab in your repository
2. Select "Daily Market Backtest" workflow
3. Click "Run workflow" button
4. Choose branch and click "Run workflow"

## Workflow Details

### What the workflow does

1. **Environment Setup**
   - Checks out repository code
   - Sets up Python 3.11
   - Installs Poetry and dependencies

2. **Run Backtest**
   - Creates data directory if needed
   - Runs `backtest.py` with environment variables
   - Fetches real-time market data
   - Generates predictions for all model combinations

3. **Commit Results**
   - Checks if new data was generated
   - Commits `data/backtest.csv` and `data/backtest_token_cost.csv`
   - Pushes changes to main branch

### Expected Output Files

- `data/backtest.csv` - Daily prediction results
- `data/backtest_token_cost.csv` - Token usage tracking

## Troubleshooting

### Common Issues

1. **Workflow fails with "Permission denied"**
   - Check repository permissions in Settings → Actions → General
   - Ensure "Read and write permissions" is enabled

2. **API Key errors**
   - Verify all four secrets are correctly set
   - Check API key validity and quotas

3. **Poetry/dependency issues**
   - Ensure `pyproject.toml` and `poetry.lock` are up to date
   - Check if all required packages are in dependencies

4. **No changes committed**
   - This is normal if backtest produces identical results
   - Check workflow logs for actual backtest execution

### Viewing Logs

1. Go to `Actions` tab
2. Click on the workflow run
3. Click on "backtest" job
4. Expand individual steps to view detailed logs

## Customization Options

### Change Schedule

Edit the cron expression in the workflow file to run at different times.

### Modify Commit Message

Edit the commit command in the workflow:

```yaml
git commit -m "Your custom message - $(date +'%Y-%m-%d')"
```

### Add Notifications

You can add notification steps (Slack, email, etc.) after the backtest completes.

### Filter Files

To commit additional files, modify the git add command:

```yaml
git add data/backtest.csv data/backtest_token_cost.csv other_file.txt
```

## Security Notes

- API keys are stored securely as GitHub secrets
- Secrets are not visible in logs or to other users
- Only repository administrators can view/edit secrets
- Workflow runs in isolated environment that's destroyed after completion

## Cost Considerations

- GitHub Actions provides 2,000 minutes/month free for public repos
- Each workflow run takes approximately 3-5 minutes
- Running daily (22 business days/month) = ~110 minutes/month
- Well within free tier limits

For private repositories, consider GitHub's pricing for additional action minutes if needed.
