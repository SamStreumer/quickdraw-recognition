"""
Interactive drawing application for Quick Draw recognition.
"""

import pygame
import numpy as np
import os
import sys
import time
import random
from pygame.locals import *
from .data_utils import convert_drawing_to_image

class DrawingApp:
    """
    Interactive drawing application with real-time recognition.
    
    Attributes:
        width (int): Width of the application window
        height (int): Height of the application window
        model: Neural network model for recognition
        categories (list): List of category names
    """
    
    def __init__(self, model=None, categories=None, width=1024, height=768):
        """
        Initialize the drawing application.
        
        Args:
            model: Neural network model for recognition
            categories (list): List of category names
            width (int): Width of the application window
            height (int): Height of the application window
        """
        # Initialize Pygame
        pygame.init()
        
        # Set up display with resizable flag
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Quick Draw Recognition")
        
        # Set up colors
        self.bg_color = (30, 30, 45)  # Deep navy blue
        self.draw_color = (255, 255, 255)  # White
        self.ui_bg_color = (40, 40, 60)  # Slightly lighter than background
        self.text_color = (220, 220, 220)  # Light gray
        self.accent_color = (94, 129, 172)  # Soft blue
        self.secondary_color = (245, 170, 66)  # Warm orange
        self.success_color = (113, 187, 117)  # Soft green
        
        # Canvas settings
        self.canvas_size = 400
        self.canvas = pygame.Surface((self.canvas_size, self.canvas_size))
        self.canvas.fill(self.bg_color)
        self.canvas_position = (50, 50)
        
        # Drawing settings
        self.brush_size = 15
        self.eraser_size = 25
        self.drawing = False
        self.erasing = False
        self.last_pos = None
        
        # Recognition settings
        self.model = model
        self.categories = categories or ["No categories loaded"]
        self.predictions = []
        self.grid_size = 28  # Size of the grid for recognition (28x28)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # UI elements
        self.font_large = pygame.font.SysFont('Arial', 32)
        self.font_medium = pygame.font.SysFont('Arial', 24)
        self.font_small = pygame.font.SysFont('Arial', 18)
        
        # Animation
        self.last_prediction_time = 0
        self.prediction_interval = 200  # ms
        self.animation_progress = 0
        self.animation_speed = 0.02
        
        # Game state
        self.current_screen = "welcome"  # "welcome", "drawing", "results"
        self.current_prompt = None
        self.score = 0
        self.correct_guesses = 0
        self.total_prompts = 0
        self.show_success_message = False
        self.success_message_time = 0
        
        # Fullscreen flag
        self.is_fullscreen = False
        
        # Initialize clock
        self.clock = pygame.time.Clock()
        
        # Generate initial prompt
        self.generate_new_prompt()
    
    def generate_new_prompt(self):
        """Generate a new drawing prompt for the user."""
        if self.categories and len(self.categories) > 1:
            self.current_prompt = random.choice(self.categories)
            self.total_prompts += 1
            self.clear_canvas()
            self.show_success_message = False
            print(f"New prompt: {self.current_prompt}")
    
    def check_prompt_success(self):
        """Check if the current drawing matches the prompt."""
        if not self.predictions or not self.current_prompt:
            return False
        
        top_prediction, probability = self.predictions[0]
        if top_prediction.lower() == self.current_prompt.lower() and probability > 50:
            if not self.show_success_message:
                self.show_success_message = True
                self.success_message_time = pygame.time.get_ticks()
                self.correct_guesses += 1
                self.score += int(probability)
            return True
        return False
    
    def update_grid_from_canvas(self):
        """
        Convert the drawing on the canvas to a 28x28 grid for recognition.
        """
        # Scale down the canvas to grid_size x grid_size
        cell_size = self.canvas_size // self.grid_size
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                # Get the corresponding region in the canvas
                region_x = x * cell_size
                region_y = y * cell_size
                
                # Sample the region
                pixel_sum = 0
                for dy in range(cell_size):
                    for dx in range(cell_size):
                        pixel = self.canvas.get_at((region_x + dx, region_y + dy))
                        pixel_sum += pixel[0]  # Just use the red channel
                
                # Average the region and normalize to 0-1
                avg = pixel_sum / (cell_size * cell_size)
                self.grid[y][x] = avg / 255.0
    
    def predict(self):
        """
        Make a prediction based on the current drawing.
        """
        if self.model is None:
            self.predictions = [(cat, 0.0) for cat in self.categories]
            return
        
        # Convert grid to the format expected by the model
        input_data = self.grid.reshape(1, -1).astype(np.float32)
        
        # Get predictions
        prediction_probs = self.model.predict(input_data)[0]
        
        # Sort predictions by probability
        self.predictions = [(self.categories[i], float(prediction_probs[i]) * 100) 
                           for i in range(len(self.categories))]
        self.predictions.sort(key=lambda x: x[1], reverse=True)
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.width, self.height = self.screen.get_size()
        else:
            self.screen = pygame.display.set_mode((1024, 768), pygame.RESIZABLE)
            self.width, self.height = 1024, 768
    
    def adjust_for_resize(self, new_width, new_height):
        """Adjust layout when window is resized."""
        self.width = new_width
        self.height = new_height
        
        # Adjust canvas position to center if window gets large enough
        if self.width > self.canvas_size + 400:  # If there's enough room
            self.canvas_position = ((self.width - self.canvas_size) // 2 - 150, 
                                   (self.height - self.canvas_size) // 2)
    
    def draw_welcome_screen(self):
        """Draw the welcome screen."""
        # Fill background
        self.screen.fill(self.bg_color)
        
        # Draw title
        title = self.font_large.render("Quick Draw Recognition", True, self.text_color)
        title_rect = title.get_rect(center=(self.width // 2, 100))
        self.screen.blit(title, title_rect)
        
        # Draw subtitle
        subtitle = self.font_medium.render("Test your drawing skills against AI!", True, self.text_color)
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, 150))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Draw instructions
        instructions = [
            "How to play:",
            "1. You'll get a prompt to draw",
            "2. Draw on the canvas using the left mouse button",
            "3. The AI will try to guess what you're drawing",
            "4. Press N for a new prompt anytime",
            "5. Press F to toggle fullscreen",
            "",
            "Press SPACE to start drawing!"
        ]
        
        y_pos = 220
        for instruction in instructions:
            text = self.font_small.render(instruction, True, self.text_color)
            text_rect = text.get_rect(center=(self.width // 2, y_pos))
            self.screen.blit(text, text_rect)
            y_pos += 30
        
        # Draw available categories
        if len(self.categories) > 1:
            category_text = self.font_small.render("Available categories:", True, self.secondary_color)
            category_rect = category_text.get_rect(center=(self.width // 2, y_pos + 20))
            self.screen.blit(category_text, category_rect)
            
            # Split categories into multiple lines
            categories_per_line = min(5, len(self.categories))
            chunks = [self.categories[i:i+categories_per_line] for i in range(0, len(self.categories), categories_per_line)]
            
            y_pos += 50
            for chunk in chunks:
                text = self.font_small.render(", ".join(chunk), True, self.text_color)
                text_rect = text.get_rect(center=(self.width // 2, y_pos))
                self.screen.blit(text, text_rect)
                y_pos += 30
    
    def draw_canvas_border(self):
        """
        Draw a border around the canvas with a 3D effect.
        """
        canvas_rect = pygame.Rect(
            self.canvas_position[0] - 10, 
            self.canvas_position[1] - 10,
            self.canvas_size + 20, 
            self.canvas_size + 20
        )
        
        # Draw outer shadow
        pygame.draw.rect(self.screen, (20, 20, 30), canvas_rect, border_radius=5)
        
        # Draw inner border
        pygame.draw.rect(self.screen, (50, 50, 70), 
                       (self.canvas_position[0] - 5, 
                        self.canvas_position[1] - 5,
                        self.canvas_size + 10, 
                        self.canvas_size + 10), 
                       border_radius=3)
    
    def draw_prompt_info(self):
        """Draw the current prompt."""
        if not self.current_prompt:
            return
            
        # Draw prompt with subtle background
        prompt_y = 15
        prompt_width = 300
        prompt_height = 40
        prompt_x = (self.width - prompt_width) // 2
        
        # Draw prompt background
        prompt_rect = pygame.Rect(prompt_x, prompt_y, prompt_width, prompt_height)
        pygame.draw.rect(self.screen, self.ui_bg_color, prompt_rect, border_radius=5)
        
        # Draw prompt text
        prompt_text = self.font_medium.render(f"Draw: {self.current_prompt}", True, self.secondary_color)
        prompt_text_rect = prompt_text.get_rect(center=(self.width // 2, prompt_y + prompt_height // 2))
        self.screen.blit(prompt_text, prompt_text_rect)
    
    def draw_predictions(self):
        """
        Draw the prediction results on the screen.
        """
        # Prediction panel position
        panel_x = self.canvas_position[0] + self.canvas_size + 50
        panel_y = self.canvas_position[1]
        panel_width = min(300, self.width - panel_x - 20)
        panel_height = self.canvas_size
        
        # Check if panel is off-screen, adjust if needed
        if panel_x + panel_width > self.width:
            panel_x = self.width - panel_width - 20
        
        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.ui_bg_color, panel_rect, border_radius=5)
        
        # Draw title
        title = self.font_large.render("Predictions", True, self.text_color)
        title_rect = title.get_rect(midtop=(panel_x + panel_width//2, panel_y + 15))
        self.screen.blit(title, title_rect)
        
        # Draw divider
        pygame.draw.line(self.screen, self.accent_color, 
                        (panel_x + 20, panel_y + 60), 
                        (panel_x + panel_width - 20, panel_y + 60), 2)
        
        # Draw predictions
        y_offset = panel_y + 80
        max_display = min(6, len(self.predictions))
        
        for i in range(max_display):
            category, prob = self.predictions[i]
            
            # Calculate bar width based on probability
            max_bar_width = panel_width - 100
            bar_width = int(max_bar_width * min(prob / 100, 1.0))
            
            # Determine text color based on match with prompt
            text_color = self.secondary_color if category.lower() == self.current_prompt.lower() else self.text_color
            
            # Draw category name
            category_text = self.font_medium.render(category, True, text_color)
            self.screen.blit(category_text, (panel_x + 20, y_offset))
            
            # Draw probability bar background
            bar_height = 24
            bar_rect = pygame.Rect(panel_x + 20, y_offset + 30, max_bar_width, bar_height)
            pygame.draw.rect(self.screen, (50, 50, 70), bar_rect, border_radius=3)
            
            # Animate bar for top prediction
            if i == 0:
                actual_width = int(bar_width * self.animation_progress)
                self.animation_progress = min(1.0, self.animation_progress + self.animation_speed)
            else:
                actual_width = bar_width
            
            # Draw filled bar
            if actual_width > 0:
                fill_color = self.secondary_color if category.lower() == self.current_prompt.lower() else self.accent_color
                fill_rect = pygame.Rect(panel_x + 20, y_offset + 30, actual_width, bar_height)
                pygame.draw.rect(self.screen, fill_color, fill_rect, border_radius=3)
            
            # Draw percentage
            percentage_text = self.font_medium.render(f"{prob:.1f}%", True, self.text_color)
            self.screen.blit(percentage_text, 
                           (panel_x + 30 + max_bar_width, y_offset + 30))
            
            y_offset += 60
            
        # Check if the current drawing matches the prompt
        success = self.check_prompt_success()
        if success and self.show_success_message:
            # Show success message
            elapsed = pygame.time.get_ticks() - self.success_message_time
            if elapsed < 5000:  # Show message for 5 seconds
                success_text = self.font_medium.render("Great job! Press N for next prompt", True, self.success_color)
                success_rect = success_text.get_rect(center=(panel_x + panel_width//2, panel_y + panel_height - 40))
                self.screen.blit(success_text, success_rect)
    
    def draw_control_info(self):
        """
        Draw minimal control info at the bottom of the screen.
        """
        # Simple control bar at the bottom
        controls_y = self.height - 40
        control_text = self.font_small.render("Left: Draw | Right: Erase | C: Clear | N: New Prompt | F: Fullscreen | Q: Quit", True, self.text_color)
        control_rect = control_text.get_rect(center=(self.width // 2, controls_y))
        self.screen.blit(control_text, control_rect)
    
    def save_screenshot(self):
        """
        Save a screenshot of the current application.
        """
        # Create screenshots directory if it doesn't exist
        os.makedirs("screenshots", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/quickdraw_{timestamp}.png"
        
        # Save screenshot
        pygame.image.save(self.screen, filename)
        print(f"Screenshot saved as {filename}")
    
    def clear_canvas(self):
        """
        Clear the drawing canvas.
        """
        self.canvas.fill(self.bg_color)
        self.grid.fill(0.0)
        self.predictions = [(cat, 0.0) for cat in self.categories]
        self.animation_progress = 0
    
    def handle_events(self):
        """
        Handle pygame events.
        
        Returns:
            bool: False if the application should quit, True otherwise
        """
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            # Key events
            elif event.type == KEYDOWN:
                if event.key == K_q:  # Quit
                    return False
                elif event.key == K_c:  # Clear canvas
                    self.clear_canvas()
                elif event.key == K_s:  # Save screenshot
                    self.save_screenshot()
                elif event.key == K_f:  # Toggle fullscreen
                    self.toggle_fullscreen()
                elif event.key == K_n:  # New prompt
                    self.generate_new_prompt()
                elif event.key == K_SPACE and self.current_screen == "welcome":
                    self.current_screen = "drawing"
                    self.generate_new_prompt()
            
            # Window resize event
            elif event.type == VIDEORESIZE:
                if not self.is_fullscreen:
                    self.adjust_for_resize(event.w, event.h)
            
            # Mouse button events
            elif event.type == MOUSEBUTTONDOWN:
                if self.current_screen == "drawing":
                    if event.button == 1:  # Left mouse button (draw)
                        self.drawing = True
                        self.erasing = False
                    elif event.button == 3:  # Right mouse button (erase)
                        self.drawing = False
                        self.erasing = True
            
            elif event.type == MOUSEBUTTONUP:
                if self.current_screen == "drawing":
                    self.drawing = False
                    self.erasing = False
                    self.last_pos = None
                    
                    # Update grid and make a new prediction
                    self.update_grid_from_canvas()
                    self.predict()
                    self.animation_progress = 0  # Reset animation
            
            # Mouse motion events
            elif event.type == MOUSEMOTION:
                if self.current_screen == "drawing" and (self.drawing or self.erasing):
                    mouse_pos = event.pos
                    canvas_x = mouse_pos[0] - self.canvas_position[0]
                    canvas_y = mouse_pos[1] - self.canvas_position[1]
                    
                    # Only draw within the canvas
                    if (0 <= canvas_x < self.canvas_size and 
                        0 <= canvas_y < self.canvas_size):
                        
                        # Choose size based on drawing or erasing
                        size = self.brush_size if self.drawing else self.eraser_size
                        color = self.draw_color if self.drawing else self.bg_color
                        
                        # Draw a circle at the current position
                        pygame.draw.circle(self.canvas, color, (canvas_x, canvas_y), size)
                        
                        # Connect to previous position if available
                        if self.last_pos:
                            last_canvas_x = self.last_pos[0] - self.canvas_position[0]
                            last_canvas_y = self.last_pos[1] - self.canvas_position[1]
                            
                            # Draw a line from the last position to the current position
                            pygame.draw.line(self.canvas, color, 
                                           (last_canvas_x, last_canvas_y), 
                                           (canvas_x, canvas_y), size * 2)
                        
                        self.last_pos = mouse_pos
                        
                        # Periodically update prediction while drawing
                        current_time = pygame.time.get_ticks()
                        if current_time - self.last_prediction_time > self.prediction_interval:
                            self.update_grid_from_canvas()
                            self.predict()
                            self.last_prediction_time = current_time
                            self.animation_progress = 0  # Reset animation
        
        return True
    
    def run(self):
        """
        Run the application main loop.
        """
        running = True
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Clear screen
            self.screen.fill(self.bg_color)
            
            # Draw current screen
            if self.current_screen == "welcome":
                self.draw_welcome_screen()
            elif self.current_screen == "drawing":
                # Draw elements
                self.draw_canvas_border()
                self.screen.blit(self.canvas, self.canvas_position)
                self.draw_prompt_info()
                self.draw_predictions()
                self.draw_control_info()
            
            # Update display
            pygame.display.flip()
            
            # Control frame rate
            self.clock.tick(60)
        
        pygame.quit()