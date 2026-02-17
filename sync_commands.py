"""Manually sync Discord slash commands."""

import asyncio
import os
from pathlib import Path

import discord
from discord.ext import commands
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Get token
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Import commands
import sys
sys.path.insert(0, str(Path(__file__).parent))
from discord_bot.commands import VoiceBotCommands


async def sync_commands():
    """Sync commands to Discord."""
    # Create minimal bot
    intents = discord.Intents.default()
    intents.message_content = True

    bot = commands.Bot(command_prefix="/", intents=intents)

    @bot.event
    async def on_ready():
        print(f"Logged in as {bot.user}")
        print(f"Connected to {len(bot.guilds)} guilds")

        # Add commands
        cmd_group = VoiceBotCommands(bot)
        bot.tree.add_command(cmd_group)

        print("Syncing commands...")
        synced = await bot.tree.sync()
        print(f"✓ Synced {len(synced)} commands to Discord!")

        # Print command names
        for cmd in synced:
            print(f"  - /{cmd.name}")

        await bot.close()

    await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(sync_commands())
