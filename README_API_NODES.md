# ComfyUI API Nodes Integration

This guide explains how to use ComfyUI API nodes in your Modal-based deployment for access to external state-of-the-art AI models.

## 🌟 What are API Nodes?

API Nodes allow you to use external AI models (like OpenAI DALL-E, Stability AI, Google Gemini) directly in your ComfyUI workflows without needing to host them locally. This gives you access to cutting-edge models while maintaining your existing workflow infrastructure.

### Supported Models Include:
- **OpenAI**: DALL-E 3, GPT-4 Vision, ChatGPT
- **Stability AI**: Stable Image Ultra, SD 3.5 Large
- **Google**: Gemini 2.5 Pro, Veo2 (video generation)
- **Black Forest Labs**: Flux 1.1 Pro Ultra
- **Others**: Luma, Runway, Ideogram, and more

## 🚀 Quick Start

### 1. Set Up ComfyUI Account

1. Create account at [https://platform.comfy.org](https://platform.comfy.org)
2. Purchase credits for API usage (prepaid system)
3. Generate an API key from your dashboard

### 2. Configure Modal Secret

Run the setup script with your API key:

```bash
python setup_api_key.py --api-key YOUR_COMFY_API_KEY_HERE
```

Or manually create the secret:

```bash
modal secret create comfyui-api-secret COMFY_API_KEY=your_api_key_here
```

### 3. Deploy with API Node Support

Your deployment now automatically includes API node support:

```bash
modal deploy comfyui_app.py
```

### 4. Use API Node Workflows

```python
# Example: Using OpenAI DALL-E 3
request_data = {
    "user_id": "user123",
    "workflow_name": "openai_dalle3",
    "parameters": {
        "prompt": "A serene mountain landscape at sunset",
        "size": "1024x1024",
        "quality": "hd",
        "style": "vivid"
    }
}

# Send to your Modal endpoint
response = requests.post(your_modal_endpoint, json=request_data)
```

## 📋 Available Workflows

### Local Workflows (No Credits Required)
- `flux_lora`: Flux.1-dev with custom LoRA support
- `flux_dedistilled_upscaler`: Local Flux with 4K upscaling

### API Node Workflows (Credits Required)
- `openai_dalle3`: OpenAI DALL-E 3 generation ($0.04/image)
- `stability_ultra`: Stability AI Ultra ($0.08/image)

## 🔧 Workflow Configuration

API node workflows are configured in `workflows/workflow_config.json`:

```json
{
  "workflows": {
    "openai_dalle3": {
      "file": "openai_dalle3_workflow.json",
      "description": "OpenAI DALL-E 3 high-quality image generation",
      "uses_api_nodes": true,
      "estimated_cost_per_image": 0.04,
      "parameters": {
        "prompt": {
          "type": "string",
          "required": true,
          "description": "Detailed text prompt for DALL-E 3"
        },
        "size": {
          "type": "string",
          "options": ["1024x1024", "1792x1024", "1024x1792"],
          "default": "1024x1024"
        }
      }
    }
  }
}
```

## 💰 Cost Management

### Credit System
- **Prepaid credits**: No unexpected charges
- **Per-use billing**: Only pay for what you generate
- **Cost transparency**: Each workflow shows estimated cost

### Monitor Usage
- Check credits at [https://platform.comfy.org](https://platform.comfy.org)
- View usage history in your dashboard
- Set up alerts for low credit balance

### Cost Examples
- OpenAI DALL-E 3: ~$0.04 per image
- Stability AI Ultra: ~$0.08 per image
- Local models: $0.00 (only GPU costs)

## 🔐 Security Best Practices

### API Key Management
- ✅ Store in Modal secrets (encrypted)
- ✅ Use environment variables
- ❌ Never commit to version control
- ❌ Don't hardcode in source code

### Network Security
- API calls require HTTPS
- ComfyUI validates API keys server-side
- Rate limiting prevents abuse

## 🛠️ Development & Testing

### Development Server
The dev server includes API node support:

```bash
# Access your Modal dev server
# API nodes will be available in the node library
```

### Testing API Nodes
1. Start with small, simple prompts
2. Monitor credit usage during testing
3. Use local models for development when possible

### Debugging
- Check logs for API authentication issues
- Verify credits are sufficient
- Ensure ComfyUI is latest nightly version

## 📊 Monitoring & Observability

### Built-in Monitoring
The system automatically:
- ✅ Validates API keys before execution
- ✅ Checks credit balance
- ✅ Logs API node usage
- ✅ Tracks cost per generation

### Health Checks
```python
# API key validation happens automatically
# Check logs for validation status:
# "✅ API key validated and credits available"
```

## 🔄 Workflow Creation

### Adding New API Node Workflows

1. **Create workflow file** in `workflows/` directory
2. **Add configuration** to `workflow_config.json`
3. **Test thoroughly** with small batches
4. **Document cost estimates** for users

### Example API Node Workflow Structure
```json
{
  "1": {
    "class_type": "OpenAI DALL·E 3",
    "inputs": {
      "prompt": "A beautiful landscape",
      "size": "1024x1024",
      "quality": "hd"
    }
  },
  "2": {
    "class_type": "Save Image",
    "inputs": {
      "images": ["1", 0],
      "filename_prefix": "dalle3_output"
    }
  }
}
```

## 🚨 Troubleshooting

### Common Issues

**"API nodes not found"**
- Update ComfyUI to latest nightly version
- Restart the server after updates

**"Invalid API key"**
- Check API key in Modal secrets
- Verify key hasn't expired
- Ensure sufficient credits

**"Insufficient credits"**
- Purchase more credits at platform.comfy.org
- Check current balance in dashboard

**"Connection timeout"**
- Verify network connectivity
- Check if ComfyUI server is healthy
- Ensure HTTPS access is available

### Getting Help

1. **Check logs** for detailed error messages
2. **Verify setup** with the setup script
3. **Test with simple workflows** first
4. **Monitor credit usage** during debugging

## 🆕 Migration from Local-Only

### Gradual Migration Strategy
1. Keep existing local workflows unchanged
2. Add API node workflows as new options
3. Let users choose based on needs/budget
4. Monitor usage patterns and costs

### Backward Compatibility
- ✅ All existing workflows continue to work
- ✅ No breaking changes to API
- ✅ Local models remain default
- ✅ API nodes are opt-in only

## 📈 Performance Considerations

### Speed Comparison
- **Local models**: Fastest startup, consistent speed
- **API nodes**: Network latency + provider processing
- **Hybrid workflows**: Best of both worlds

### When to Use Each
- **Local**: Development, high-volume, cost-sensitive
- **API nodes**: Latest models, specialized capabilities, testing
- **Hybrid**: Complex workflows with multiple stages

## 🔮 Future Enhancements

### Planned Features
- [ ] Automatic cost optimization
- [ ] Batch processing for API nodes
- [ ] Credit usage analytics
- [ ] Custom API node integrations
- [ ] Workflow cost estimation

### Contributing
- Report issues with API node integration
- Suggest new API provider integrations
- Share workflow templates
- Contribute to documentation

---

## Support

For technical issues:
- Check this documentation
- Review Modal logs
- Test with simplified workflows

For billing/account issues:
- Contact ComfyUI support at support@comfy.org
- Visit platform.comfy.org for account management

---

**Happy generating! 🎨** 