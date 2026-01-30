import OpenAI from 'openai';

let client = null;

function getClient() {
  if (client) return client;

  if (process.env.AZURE_OPENAI_API_KEY && process.env.AZURE_OPENAI_ENDPOINT) {
    const { AzureOpenAI } = OpenAI;
    client = new AzureOpenAI({
      apiKey: process.env.AZURE_OPENAI_API_KEY,
      endpoint: process.env.AZURE_OPENAI_ENDPOINT,
      apiVersion: process.env.AZURE_OPENAI_API_VERSION || '2024-02-01',
    });
  } else if (process.env.OPENAI_API_KEY) {
    client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  } else {
    throw new Error(
      'No API key found. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT'
    );
  }

  return client;
}

function getModel() {
  if (process.env.AZURE_OPENAI_DEPLOYMENT) {
    return process.env.AZURE_OPENAI_DEPLOYMENT;
  }
  return process.env.OPENAI_MODEL || 'gpt-4o-mini';
}

export async function chat(messages, systemPrompt) {
  const openai = getClient();
  const model = getModel();

  const fullMessages = [
    { role: 'system', content: systemPrompt },
    ...messages,
  ];

  const response = await openai.chat.completions.create({
    model,
    messages: fullMessages,
    temperature: 0.7,
  });

  return response.choices[0].message.content;
}
