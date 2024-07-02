class StoryNode:
    def __init__(self, text, branches):
        self.text = text
        self.branches = branches

class InteractiveStory:
    def __init__(self, start_node):
        self.current_node = start_node

    def tell(self):
        print(self.current_node.text)

    def choose(self, option):
        if option in self.current_node.branches.keys():
            self.current_node = self.current_node.branches[option]

# Create nodes
end_node = StoryNode("The end.", {})
option1_node = StoryNode("Option 1 leading here", {"end": end_node})
option2_node = StoryNode("Option 2 leading here", {"end": end_node})
start_node = StoryNode("Start here", {"option1": option1_node, "option2": option2_node})
story = InteractiveStory(start_node)

# Tell the story
story.tell()
story.choose("option1")
story.tell()
story.choose("end")
story.tell()