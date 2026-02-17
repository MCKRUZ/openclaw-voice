"""Quick command sync script."""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import discord
from dotenv import load_dotenv
from discord_bot.commands import VoiceBotCommands

load_dotenv()

async def main():
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    tree = discord.app_commands.CommandTree(client)

    @client.event
    async def on_ready():
        print(f"Connected as {client.user}")

        # Add command group
        commands = VoiceBotCommands(client)
        tree.add_command(commands)

        # Sync
        print("Syncing commands to Discord...")
        synced = await tree.sync()

        print(f"SUCCESS! Synced {len(synced)} command(s):")
        for cmd in synced:
            print(f"  /{cmd.name}")

        await client.close()

    try:
        await client.start(os.getenv("DISCORD_TOKEN"))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())
