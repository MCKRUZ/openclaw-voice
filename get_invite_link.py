"""Generate proper invite link with slash command permissions."""
import asyncio
import os
from dotenv import load_dotenv
import discord

load_dotenv()

async def main():
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"\nBot: {client.user.name}")
        print(f"Bot ID: {client.user.id}")
        print(f"\n{'='*70}")
        print("REINVITE LINK (with slash command permissions):")
        print('='*70)

        # Create invite URL with proper permissions
        permissions = discord.Permissions(
            connect=True,
            speak=True,
            use_voice_activation=True,
            send_messages=True,
            read_messages=True,
            view_channel=True,
        )

        url = discord.utils.oauth_url(
            client.user.id,
            permissions=permissions,
            scopes=["bot", "applications.commands"]
        )

        print(f"\n{url}\n")
        print("="*70)
        print("\nInstructions:")
        print("1. Click the link above")
        print("2. Select your server")
        print("3. Authorize the bot")
        print("4. Slash commands will work immediately!")
        print("="*70)

        await client.close()

    await client.start(os.getenv("DISCORD_TOKEN"))

if __name__ == "__main__":
    asyncio.run(main())
