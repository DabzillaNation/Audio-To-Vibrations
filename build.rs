    use std::{env, io};
    use winresource::WindowsResource;

    fn main() -> io::Result<()> {
        if env::var_os("CARGO_CFG_WINDOWS").is_some() {
            WindowsResource::new()
                // Specify the path to your .ico file relative to the crate root.
                .set_icon("assets/vibrator.ico") 
                .compile()?;
        }
        Ok(())
    }