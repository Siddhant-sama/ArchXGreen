#!/bin/bash
# Build and push ArchXGreen Docker images to GitHub Container Registry (GHCR)
# Usage: ./push-to-ghcr.sh [green|purple|all]

set -e

REGISTRY="ghcr.io"
OWNER="siddhant-sama"
REPO="archxgreen"
GREEN_IMAGE="$REGISTRY/$OWNER/$REPO:latest"
PURPLE_IMAGE="$REGISTRY/$OWNER/$REPO/purple:latest"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== ArchXGreen Docker Image Build & Push ===${NC}"

# Check if GitHub token is available
if [ -z "$GITHUB_TOKEN" ]; then
    echo -e "${BLUE}GitHub token not found. Please set GITHUB_TOKEN environment variable.${NC}"
    echo "Or login manually: docker login ghcr.io -u <username> -p <token>"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Determine which images to build
BUILD_GREEN=false
BUILD_PURPLE=false

case "${1:-all}" in
    green)
        BUILD_GREEN=true
        ;;
    purple)
        BUILD_PURPLE=true
        ;;
    all)
        BUILD_GREEN=true
        BUILD_PURPLE=true
        ;;
    *)
        echo "Usage: $0 [green|purple|all]"
        exit 1
        ;;
esac

# Build and push green agent
if [ "$BUILD_GREEN" = true ]; then
    echo -e "${BLUE}Building green agent image: $GREEN_IMAGE${NC}"
    docker build -t "$GREEN_IMAGE" .
    
    echo -e "${BLUE}Pushing green agent image...${NC}"
    docker push "$GREEN_IMAGE"
    echo -e "${GREEN}✓ Green agent image pushed: $GREEN_IMAGE${NC}"
fi

# Build and push purple agent
if [ "$BUILD_PURPLE" = true ]; then
    echo -e "${BLUE}Building purple agent image: $PURPLE_IMAGE${NC}"
    docker build -t "$PURPLE_IMAGE" -f Dockerfile.purple .
    
    echo -e "${BLUE}Pushing purple agent image...${NC}"
    docker push "$PURPLE_IMAGE"
    echo -e "${GREEN}✓ Purple agent image pushed: $PURPLE_IMAGE${NC}"
fi

echo -e "${GREEN}=== All Done! ===${NC}"
echo ""
echo "Next steps:"
echo "1. Register green agent at: https://agentbeats.dev/"
echo "   Image URL: $GREEN_IMAGE"
echo "2. Register baseline purple agent at: https://agentbeats.dev/"
echo "   Image URL: $PURPLE_IMAGE"
echo "3. Save the AgentBeats IDs and update README.md with Registration section"
