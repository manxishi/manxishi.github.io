---
title: "Connect with Me"
---

<div class="connect-content">
    <p>Feel free to reach out! You can connect with me on <a href="https://www.linkedin.com/in/manxishi/" target="_blank" rel="noopener noreferrer">LinkedIn</a> or use the contact form below.</p>

    <div class="contact-form-wrapper">
        <h2>Contact Form</h2>
        <form action="https://formspree.io/f/mkgwalkl" method="POST">
            <input type="text" name="name" placeholder="Your Name" required>
            <input type="email" name="email" placeholder="Your Email" required>
            <textarea name="message" rows="5" placeholder="Your Message"></textarea>
            <button type="submit">Send Message</button>
        </form>
    </div>
</div>

<style>
.connect-content {
    max-width: 600px;
    margin: 0 auto;
    text-align: center;
}

.connect-content > p {
    margin-bottom: 2rem;
    font-size: 1.1rem;
}

.contact-form-wrapper {
    background: var(--theme);
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.contact-form-wrapper h2 {
    margin-top: 0;
    margin-bottom: 1.5rem;
    font-size: 1.5rem;
}

.contact-form-wrapper form {
    display: flex;
    flex-direction: column;
    max-width: 100%;
    margin: 0 auto;
    text-align: left;
}

.contact-form-wrapper form input,
.contact-form-wrapper form textarea {
    width: 100%;
    margin-bottom: 1rem;
    padding: 0.75rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 1rem;
    font-family: inherit;
    background: var(--theme);
    box-sizing: border-box;
}

.contact-form-wrapper form textarea {
    resize: vertical;
    min-height: 120px;
}

.contact-form-wrapper form button {
    width: 100%;
    padding: 0.75rem 1.5rem;
    background-color: var(--primary);
    color: var(--theme);
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.3s ease;
    margin-top: 0.5rem;
}

.contact-form-wrapper form button:hover {
    opacity: 0.8;
}

@media (max-width: 600px) {
    .contact-form-wrapper {
        padding: 1.5rem;
    }
}
</style>

