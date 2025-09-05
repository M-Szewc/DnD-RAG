import type { PropsWithChildren } from "react";

const Layout = ({ children }: PropsWithChildren) => {
    return (
    <div>
        <header>
            This is rag chat
        </header>
        <main>
            {children}
        </main>
        <footer>
            <div>
                <p>This is footer</p>
            </div>
        </footer>
    </div>
    );
};

export default Layout;